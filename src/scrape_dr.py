"""Parse and clean tabular data on U.S. capital punishment.

Source: Office of the Clark County (Indiana) Prosecuting Attorney
        http://www.clarkprosecutor.org/html/death/usexecute.htm

Notes:
- The page uses some structurally nonsensical HTML; the normal
  <table>, <tr>, <td> hierarchy is nonexistent here and instead we
  have <trs> within <td>s within <tds>, and so on.
- This, coupled with the need to *treat "separating" nested tags like
  <br> as "structural delimiters," makes this a less than ideal case
  for using `pandas.read_html()`.  Instead, I build a generator
  and "chunk" it into fixed-length blocks representing rows.  It is
  a fairly fast and sturdy alternative.
"""


from collections import OrderedDict
from datetime import datetime
from functools import reduce
from itertools import zip_longest, product
import re
from string import whitespace
import warnings

from bs4 import BeautifulSoup, SoupStrainer, NavigableString
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import signal

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, RFE
from sklearn.linear_model import (LinearRegression,
                                  Lasso,
                                  Ridge,
                                  ElasticNet)
from sklearn.model_selection import PredefinedSplit, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# DEMOGRAPHIC = re.compile(r'(\w+|\?+)\s*\/\s*(\w+|\?+)\s*\/\s*(\w+|\?+)')
NCOLUMNS = 11
URL = 'http://www.clarkprosecutor.org/html/death/usexecute.htm'
STRAINER = SoupStrainer('table', attrs={'width': '100%'})
RANDOM_STATE = 444
np.random.seed(RANDOM_STATE)


def tagfilter(tag):
    """Passed to `soup.find_all()` as callable."""
    return tag.name == 'td' and len(tag.contents) < 3 and tag.find('font')


def parse_broken_tree():
    # SoupStrainer not compatible with html5lib
    soup = BeautifulSoup(requests.get(URL).text, 'html.parser',
                         parse_only=STRAINER)
    for table in soup.find_all('table', attrs={'width': '100%'}):
        cells = table.find_all(tagfilter)
        for tag in cells:
            txt = tag.find('font')  # attrs={'face': 'arial'}
            if txt.find('td'):
                container = []
                # We have an unclosed <td>
                center = txt.find('center')
                for i in center.descendants:
                    if isinstance(i, NavigableString) and i not in whitespace and i:  # noqa
                        container.append(i.string.strip())
                yield container
            else:
                container = []
                for i in txt.descendants:
                    if isinstance(i, NavigableString) and i not in whitespace and i:  # noqa
                        container.append(i.string.strip())
                yield container


def chunk(obj, n, padvalue=None):
    """Collect data into fixed-length chunks or blocks."""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    # Source:
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    return zip_longest(*[iter(obj)] * n, fillvalue=padvalue)


# We're working with 11 columns.  Having lost the <table>/<tr>/<td>
#     structure, this hack actually works well.
data = list(chunk(parse_broken_tree(), NCOLUMNS))

# Cleaning
# ---------------------------------------------------------------------


def centurize(dt):
    if dt > datetime.now():
        dt -= relativedelta(years=100)
    return dt


def parse_broken_date(date_string, fmt='%m-%d-%y'):
    if date_string == '02-29-01':
        # One typo
        return datetime(2001, 2, 28)
    if 'to' in date_string:
        # Another typo
        return 'NaN'
    try:
        return centurize(datetime.strptime(date_string, fmt))
    except ValueError:
        try:
            month, day, year = date_string.split('-')
            if len(year) == 4:
                # '06-03-1990'
                year = year[-2:]
        except ValueError:
            # Handle cases like '1992' (no month/day given)
            date_string = '-'.join(('01', '01', date_string[-2:]))
            return centurize(datetime.strptime(date_string, fmt))
        if '?' in year:
            # We can't really do anything here
            # Deliberately use string-form NaN here (for now)
            return 'NaN'
        if '?' in month or month == '00' or month == '0':
            month = '01'
        if '?' in day or day == '00' or day == '0':
            day = '01'
        date_string = '-'.join((month, day, year))
        return centurize(datetime.strptime(date_string, fmt))


def fix_demographic(demo, type_='victim'):
    # W / M / ?? --> W / M / NaN
    # Keep as str NaN rather than np.nan for now
    try:
        race, gender, age = re.split('\s*/\s*', re.sub('\?+', 'NaN', demo))
    except ValueError:
        race, gender, age = 'NaN', 'NaN', 'NaN'
    if type_ == 'murderer':
        if ',' in age:
            age, *_ = age.split(',')
        elif '-' in age:
            age, *_ = re.split('\s?-\s?', age)
    return dict(race=race, gender=gender, age=age)


def find_triplets(info, type_='victim'):
    # https://stackoverflow.com/a/48362451/7954504
    info = iter(info)
    while True:
        try:
            name = next(info)
        except StopIteration:
            # https://www.python.org/dev/peps/pep-0479/
            return
        try:
            demo = next(info)
        except StopIteration:
            return
        officer = demo.upper() == 'OFFICER'
        if officer:
            demo = next(info)
        yield name, officer, demo


def parse_info(info, type_='victim'):
    if type_ == 'victim':
        # List of OrderedDict
        return list(OrderedDict(name=name, officer=officer,
                                **fix_demographic(demo))
                    for name, officer, demo in find_triplets(info))
    elif type_ == 'murderer':
        # Single-level dictionary--one name
        name, *_, demo = info
        return OrderedDict(name=name,
                           **fix_demographic(demo, type_='murderer'))


def reorder_product(*a):
    for tup in product(*a[::-1]):
        yield tup[::-1]


def expand_info(info, type_='victim', name=None):
    fields = ['name', 'officer', 'race', 'gender', 'age']
    n = len(info)
    numbers = (str(i) for i in range(1, n + 1))
    columns = [*(''.join(t) for t in reorder_product(fields, numbers))]
    # TODO: will need to pass a `name`
    df = pd.DataFrame(info).stack().to_frame(name=name).T
    df.columns = columns
    return df


def merge_info(dfs):
    # `dfs` are assume to be "expanded"
    # Not worrying about unsortedness of columns right now
    return reduce(lambda left, right: pd.concat((left, right)), dfs)


# Filter out header rows
data = [row for row in data if row[1][0] != 'Date of Execution']

# Unzip individual fields
states, methods = zip(*[x[2:4] for x in data])
state = [st[0] for st in states]
method_of_execution = [method[0] for method in methods]
date_of_execution = [parse_broken_date(info[1][0]) for info in data]
date_of_birth = [parse_broken_date(info[5][0]) for info in data]

# Multiple murders: use *least* recent (time bias)
date_of_murder = []
for info in data:
    dates = (parse_broken_date(date) for date in info[7])
    try:
        minimum = min(date for date in dates
                      if isinstance(date, datetime))
    except ValueError:  # empty
        minimum = 'NaN'
    date_of_murder.append(minimum)

# Same with sentencing - earliest first
date_of_sentence = []
for info in data:
    dates = (parse_broken_date(date) for date in info[-1]
             if not date.replace(' ', '').isalpha())
    try:
        minimum = min(date for date in dates
                      if isinstance(date, datetime))
    except ValueError:  # empty
        minimum = 'NaN'
    date_of_sentence.append(minimum)

method_of_murder = [';'.join([i.replace(',', '').lower()
                    for i in info[8]]) for info in data]
relationship = [';'.join([i.replace(',', '').lower()
                for i in info[9]]) for info in data]

master = OrderedDict(
    state=state,
    method_of_execution=method_of_execution,
    date_of_execution=date_of_execution,
    date_of_birth=date_of_birth,
    date_of_murder=date_of_murder,
    date_of_sentence=date_of_sentence,
    method_of_murder=method_of_murder,
    relationship=relationship)
master = pd.DataFrame(master)

victim = merge_info([expand_info(parse_info(info[6])) for info in data])
murderer = pd.DataFrame([parse_info(info[4], type_='murderer')
                         for info in data]).add_prefix('murderer_')

master = master.merge(victim.reset_index(drop=True), left_index=True,
                      right_index=True)
master = master.merge(murderer, left_index=True, right_index=True)\
    .sort_values('date_of_sentence')
print('Original shape of raw data:', master.shape)

# Additional cleanup
# ---------------------------------------------------------------------
cols = master.filter(regex='gender\d').columns
master.loc[:, cols] = master.loc[:, cols].replace('W', 'F')

# McVeigh = 168 victims with no specific info
# http://www.clarkprosecutor.org/html/death/US/mcveigh717.htm
master = master[master['murderer_name'] != 'Timothy James McVeigh']

cols = master.filter(regex='race\d').columns
mapping = {
    'A': 'asian',
    'B': 'black',
    'H': 'hispanic',
    'I': 'indian',
    'W': 'white',
    'Korean': 'asian',
    'Portugese': 'hispanic',
    'Iraqi': 'arab',
    'Pakistani': 'arab',
    'NA': 'native_american'
    }
master.loc[:, cols] = master.loc[:, cols]\
    .applymap(lambda x: mapping.get(x, x))\
    .apply(lambda col: col.str.lower())\
    .fillna('nan')

master['murderer_race'] = master['murderer_race'].map(mapping)

cols = master.filter(regex='age\d').columns
master.loc[:, cols] = master.loc[:, cols].apply(lambda col: pd.to_numeric(
    col, errors='coerce'))

# Dtype cleanup
# ---------------------------------------------------------------------

cols = master.filter(regex='officer\d').columns
master.loc[:, cols] = master.loc[:, cols].astype(float)

categories = ['state', 'method_of_execution', 'murderer_race',
              'murderer_gender']

master.loc[:, categories] = master.loc[:, categories].apply(
    lambda col: col.astype('category'))

# This is murderer age at time of murder, not execution;
# It is a given, not derived, field
master['murderer_age'] = pd.to_numeric(master['murderer_age'],
                                       errors='raise')

# Categorize `race` (no dummification yet)
races = pd.unique(master.filter(regex='race\d').values.ravel('K'))
cols = master.filter(regex='race\d').columns
master.loc[:, cols] = master.loc[:, cols].apply(lambda col: col.astype(
    'category', categories=races))
master['white_victim_binary'] = master.loc[:, cols].eq('white')\
    .any(axis=1).astype(float)
master['minority_binary'] = master.loc[:, cols].filter(regex='race\d')\
    .ne('white').all(axis=1).astype(float)

# Categorize `gender` (no dummification yet)
g = ['m', 'f', 'nan']
cols = master.filter(regex='gender\d').columns
master.loc[:, cols] = master.loc[:, cols].fillna('nan').apply(
    lambda col: col.str.lower().astype('category', categories=g))
print('Shape after initial cleaning:', master.shape)
print('(Dropped 1 record - Timothy McVeigh)')

# Dummification
# ---------------------------------------------------------------------

dummy_cols = [
    'method_of_execution',
    'state',
    'murderer_race',
    'murderer_gender'
    ]

# Automatically drops `dummy_cols` also
master = pd.get_dummies(master, columns=dummy_cols, drop_first=True)

# Drop states with < 10 entries
mask = master.filter(like='state_').eq(1.).sum(axis=0).lt(10)
states = mask.index[mask]
master.drop(states, axis=1, inplace=True)
print('Shape after dropping states with < 10 records:', master.shape)

# New (derived) fields
# ---------------------------------------------------------------------

master['officer'] = master.filter(regex='officer\d').any(axis=1).astype(int)
master['num_officers'] = master.filter(regex='officer\d').sum(axis=1).fillna(0)
master['time_on_deathrow'] = master['date_of_execution'].sub(
    master['date_of_sentence']).dt.days
# This doesn't technically represent a trial length--just named so
master['trial_length'] = master['date_of_sentence'].sub(
    master['date_of_murder']).dt.days
master['num_victims'] = master.filter(regex='name\d').count(axis=1)
master['num_female_victims'] = master.filter(regex='gender\d').eq('f').sum(
        axis=1)
master['female_victim_binary'] = master['num_female_victims'].gt(0)\
    .astype(float)
master['median_victim_age'] = master.filter(regex='age\d').median(axis=1)
master['child_victim'] = master.filter(regex='age\d').lt(21).any(axis=1)\
    .astype(float)

classifiers = {
    'method_of_murder': [
        ('shot', 'handgun|shotgun|rifle|firearm|ak-47|ar-15|pistol|mac-'),
        ('strangled', 'strangle|strangulation|stragulation|asphyxia|smother|cord|rope|suffocat'),  # noqa
        ('stabbed', 'stabbing|knife|cut|stabbed|slash|scissors|ax|poker|screwdriver'),  # noqa
        ('beat', 'stomping|beat|bludgeon|struck|club|hit')
        ],
    'relationship': [
        ('family', 'boyfriend|husband|wife|girlfriend|fiance|son|daughter|niece|newphew|father|mother'),  # noqa
        ('friend', 'acquaintance|friend|coworker|co-worker|employe[re]|'),
        ('none', 'none'),
        ('cellmate', 'cellmate|inmate|jailmate|')
        ]
    }

for k, v in classifiers.items():
    for method, regex in v:
        master.loc[:, k + '_' + method] = master[k].str.contains(
            regex, regex=True, case=False).astype(float)
print('Shape after dummification:', master.shape)

# OLD - we don't want these fields, period
# Dates expressed as days since Jan 1, 1920
# anchor = datetime(1920, 1, 1)
# master['date_of_murder'] = master['date_of_murder'].sub(anchor).dt.days
# master['date_of_birth'] = master['date_of_birth'].sub(anchor).dt.days

keep = [
    'officer',
    'num_officers',
    'time_on_deathrow',
    'trial_length',
    'murderer_age',
    'num_victims',
    'num_female_victims',
    'female_victim_binary',
    'minority_binary',
    'white_victim_binary',
    'median_victim_age',
    'child_victim'
    ]

# _constrained = master.set_index('date_of_sentence')\
#     .truncate(before=datetime(2000, 1, 1))\
#     .reset_index()

keep += [col for col in master.columns if col.startswith((
    'state_', 'method_of_execution_', 'murderer_race_',
    'murderer_gender_', 'relationship_', 'method_of_murder_'))]
scatter = master[['date_of_sentence', 'time_on_deathrow']].copy()  # for later
master = master.loc[:, keep]

master['median_victim_age'] = master['median_victim_age'].fillna(
    master['median_victim_age'].mean())
print('Shape after manually dropping fields:', master.shape)

n = len(master)
master.dropna(inplace=True)
print('Dropped {} records due to NaN data.'.format(n - len(master)))
print('Shape after `dropna()`', master.shape)

# Visually, do we have nonstationarity in y? (yes)
# ---------------------------------------------------------------------

# Add in currently-on-death-row (Texas)
threshold = pd.datetime(2014, 12, 31)
url = 'http://www.tdcj.state.tx.us/death_row/dr_offenders_on_dr.html'
texas = pd.read_html(url, header=0, parse_dates=['Date of  Birth',
                                                 'Date  Received',
                                                 'Date of  Offense'])[0]\
    .set_index('Date  Received').truncate(before=threshold)
texas = texas.assign(
    time_on_deathrow=(texas.index - threshold).days * -1)
texas = texas[['time_on_deathrow']].copy().sort_index()
texas['status'] = 'on_deathrow'

# Add in currently-on-death-row (California)
ca = pd.read_csv('metis/metisgh/projectluther/ca.csv',
                 usecols=['date_of_sentence'],
                 parse_dates=['date_of_sentence'],
                 infer_datetime_format=True)\
    .dropna()
ca['status'] = 'on_deathrow'
ca.set_index('date_of_sentence', inplace=True)
ca['time_on_deathrow'] = (ca.index - threshold).days * -1
ca = ca.sort_index().truncate(after=threshold)

on_deathrow = pd.concat((texas, ca)).sort_index()
assert scatter['date_of_sentence'].dropna().is_monotonic_increasing
scatter = scatter.dropna().set_index('date_of_sentence')
scatter['status'] = 'executed'
merged = pd.concat((scatter, on_deathrow)).sort_index().dropna()

# First plot executed only, then combine

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(merged[merged.status == 'on_deathrow']['time_on_deathrow'],
        color='g', linestyle='', label='On Death Row')
ax.plot(merged[merged.status == 'executed']['time_on_deathrow'],
        color='b', linestyle='', label='Executed')
ax.legend()
ax.set_ylabel('Days on Death Row')
ax.set_xlabel('Date of Sentence')
ax.set_title('Time on Death Row as a Function of Sentence Date')
fig.savefig('metis/metisgh/projectluther/time_on_deathrow_merged.png')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(merged[merged.status == 'executed']['time_on_deathrow'],
        color='b', linestyle='', label='Executed')
ax.legend()
ax.set_ylabel('Days on Death Row')
ax.set_xlabel('Date of Sentence')
ax.set_title('Time on Death Row as a Function of Sentence Date')
fig.savefig('metis/metisgh/projectluther/time_on_deathrow.png')

# Preprocessing/feature selection
# ---------------------------------------------------------------------

y = master.pop('time_on_deathrow')

# Detrend y - it is almost perfectly linear and benefits well
#    from linear de-trending.
# TODO: we will need to transform back later for prediction/testing
_y = y.copy()
y = signal.detrend(data=y)

# We still have some extremities early in the time series despite de-trending
#    (10-15 or so).  Drop 2-sigma outliers.
mask = np.less(np.abs(y), y.mean() + 2 * y.std())
y = y[mask]
master = master.loc[mask]
print('Shape after outlier masking:', master.shape)
assert len(y) == len(master)

# OLD
# Remove all binary features that are either one or zero in more than
#     90% of the samples.  Var(X) = p(1-p)
# http://scikit-learn.org/stable/modules/feature_selection.html
#     #removing-features-with-low-variance
# Easier to do directly in pandas than with sklearn.VarianceThreshold
# n = master.shape[1]
# threshold = 0.9 * (1. - 0.9)
# dummy_cols = master.columns[(master.eq(0.) | master.eq(1.)).all(axis=0)]
# low_var = master[dummy_cols].var().lt(threshold)
# low_var = low_var[low_var].index
# master.drop(low_var, axis=1, inplace=True)
# # TODO: this drops `officer`
# print('Dropped {} features due to variance threshold.'.format(
#     n - master.shape[1]))

# Univariate feature selection
# ---------------------------------------------------------------------

# Mimic SelectKBest(score_func=f_regression, percentile=50)...
# But we want to (a) threshold on a p value and (b) retain feature names
fvals, pvals = f_regression(X=master, y=y, center=False)
threshold = 0.30
f = pd.DataFrame({'fval': fvals, 'pvals': pvals},
                 index=master.columns).sort_values('pvals')
f.to_csv('metis/metisgh/projectluther/f_regrs.csv')
n = master.shape[1]
print('Dropped {} features in feature selection.'.format(n - master.shape[1]))
print('New design matrix shape:', master.shape)


# Using PredefinedSplit (may make sense with time series)
# ---------------------------------------------------------------------

threshold = 650
X_train, X_test = np.array_split(master, [threshold])
y_train, y_test = np.array_split(y, [threshold])

# Indices with -1 kept in train; indices >= 0 put in validation
# https://stackoverflow.com/a/48398977/7954504
# So we now have:
# - training data: length 500
# - validation data: length 150
# - test data: the remainder/tail end (about 110 records)
train_idx = np.full((500,), -1, dtype=np.int_)
test_idx = np.full((150,), 0, dtype=np.int_)

fold = np.append(train_idx, test_idx)
ps = PredefinedSplit(fold)

# We will look at: linear regression, lasso, ridge, knn regression,
#     elasticnet, decision tree regression, random forest regression--
#     and using a grid search for all but linear regression.

knn_params = {'knn__n_neighbors': [1, 3, 5, 7]}
ridge_params = {'ridge__alpha': np.linspace(0.1, 1.0, num=10)}
lasso_params = {'lasso__alpha': np.linspace(0.1, 1.0, num=10)}
net_params = {'net__alpha': np.linspace(0.1, 1.0, num=10),
              'net__l1_ratio': np.linspace(0., 1., num=5)}
tree_params = {'tree__max_depth': np.linspace(5, 25, num=5, dtype=np.int_),
               'tree__max_features': np.arange(3, 11, dtype=np.int_)}
forest_params = {'forest__max_depth': np.linspace(5, 25, num=5, dtype=np.int_),
                 'forest__max_features': np.arange(3, 11, dtype=np.int_)}

warnings.filterwarnings('ignore')

# Standard linear regression
linear = LinearRegression(n_jobs=-1).fit(X=X_train, y=y_train)
print(linear.__class__.__name__, '\n', '=' * 15, sep='')
print('Training score: {:0.2f}'.format(linear.score(X=X_train,
                                                    y=y_train)))
print('Test set score: {:0.2f}'.format(linear.score(X=X_test, y=y_test)))
print()


# k-neighbors regression
pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
grid_search = GridSearchCV(pipe, knn_params, cv=ps)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['knn'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print()


# Ridge regression (L2 regularization)
# Increasing alpha forces coefficients to move towards zero;
#     might help generalization at expense of training accuracy
pipe = Pipeline([('scaler', StandardScaler()),
                 ('ridge', Ridge(random_state=RANDOM_STATE, max_iter=10000))])
grid_search = GridSearchCV(pipe, ridge_params, cv=ps)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['ridge'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print()


# Lasso regression (L1 regularization)
# Forces coefficients to 0
pipe = Pipeline([('scaler', StandardScaler()),
                 ('lasso', Lasso(random_state=RANDOM_STATE, max_iter=10000))])
grid_search = GridSearchCV(pipe, lasso_params, cv=ps)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['lasso'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Number of features used: {}\n\n'.format(
    np.sum(grid_search.best_estimator_.named_steps['lasso'].coef_ != 0)))


# ElasticNet Regression
# `alpha` --> [0, 1]
# `l1_ratio` --> [0, 1]
# For `l1_ratio` = 0 the penalty is an L2 penalty
# For `l1_ratio` = 1 it is an L1 penalty
pipe = Pipeline([('scaler', StandardScaler()),
                 ('net', ElasticNet(random_state=RANDOM_STATE,
                                    max_iter=20000))])
grid_search = GridSearchCV(pipe, net_params, cv=ps)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['net'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Number of features used: {}\n\n'.format(
    np.sum(grid_search.best_estimator_.named_steps['net'].coef_ != 0)))


# Decision tree regression
# max_depth=4 --> only 4 "questions" can be asked
pipe = Pipeline([('scaler', StandardScaler()),
                 ('tree', DecisionTreeRegressor(random_state=RANDOM_STATE))])
grid_search = GridSearchCV(pipe, tree_params, cv=ps)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['tree'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Feature importances:\n{}\n\n'.format(
    grid_search.best_estimator_.named_steps['tree'].feature_importances_),
    sep='')
print()


# Random forest regression(collection of decision trees)
# n_estimators --> number of trees; larger=better
pipe = Pipeline([('scaler', StandardScaler()),
                 ('forest', RandomForestRegressor(random_state=RANDOM_STATE,
                                                  n_estimators=1500))])
grid_search = GridSearchCV(pipe, forest_params, cv=ps)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['forest'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Feature importances:\n{}\n\n'.format(
    grid_search.best_estimator_.named_steps['forest'].feature_importances_),
    sep='')
print()


plt.matshow(grid_search.cv_results_['mean_test_score'].reshape(8, -1),
            vmin=-2.5, vmax=2.5, cmap="RdYlGn")
plt.xlabel("forest__max_depth")
plt.ylabel("forest__max_features")
plt.xticks(range(len(forest_params['forest__max_depth'])),
           forest_params['forest__max_depth'])
plt.yticks(range(len(forest_params['forest__max_features'])),
           forest_params['forest__max_features'])
plt.colorbar()
plt.title('Predfined train/val. split')
plt.savefig('/Users/brad/Scripts/python/metis/metisgh/projectluther/cmap.png')


# Kfold
# ---------------------------------------------------------------------
# Let's also try the above with normal cross-validation rather than
#     explicitly specifying train/validation split.
# cv=5 --> five-fold cross-validation (not stratified for regression)
# The k-folds are not shuffled by default; let's shuffle them
# Note we still need the original train/test here.

kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# k-neighbors regression
pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
grid_search = GridSearchCV(pipe, knn_params, cv=kfold)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['knn'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print()


# Ridge regression (L2 regularization)
# Increasing alpha forces coefficients to move towards zero;
#     might help generalization at expense of training accuracy
pipe = Pipeline([('scaler', StandardScaler()),
                 ('ridge', Ridge(random_state=RANDOM_STATE, max_iter=10000))])
grid_search = GridSearchCV(pipe, ridge_params, cv=kfold)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['ridge'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print()


# Lasso regression (L1 regularization)
# Forces coefficients to 0
pipe = Pipeline([('scaler', StandardScaler()),
                 ('lasso', Lasso(random_state=RANDOM_STATE, max_iter=10000))])
grid_search = GridSearchCV(pipe, lasso_params, cv=kfold)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['lasso'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Number of features used: {}\n\n'.format(
    np.sum(grid_search.best_estimator_.named_steps['lasso'].coef_ != 0)))


# ElasticNet Regression
# `alpha` --> [0, 1]
# `l1_ratio` --> [0, 1]
# For `l1_ratio` = 0 the penalty is an L2 penalty
# For `l1_ratio` = 1 it is an L1 penalty
pipe = Pipeline([('scaler', StandardScaler()),
                 ('net', ElasticNet(random_state=RANDOM_STATE,
                                    max_iter=20000))])
grid_search = GridSearchCV(pipe, net_params, cv=kfold)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['net'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Number of features used: {}\n\n'.format(
    np.sum(grid_search.best_estimator_.named_steps['net'].coef_ != 0)))


# Decision tree regression
# max_depth=4 --> only 4 "questions" can be asked
pipe = Pipeline([('scaler', StandardScaler()),
                 ('tree', DecisionTreeRegressor(random_state=RANDOM_STATE))])
grid_search = GridSearchCV(pipe, tree_params, cv=kfold)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['tree'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Feature importances:\n{}\n\n'.format(
    grid_search.best_estimator_.named_steps['tree'].feature_importances_),
    sep='')
print()


# Random forest regression(collection of decision trees)
# n_estimators --> number of trees; larger=better
pipe = Pipeline([('scaler', StandardScaler()),
                 ('forest', RandomForestRegressor(random_state=RANDOM_STATE,
                                                  n_estimators=1500))])
grid_search = GridSearchCV(pipe, forest_params, cv=kfold)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_.named_steps['forest'].__class__.__name__,
      '\n', '=' * 15, sep='')
print('Mean cross-validation score: {:.2f}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print('Test set score: {:.2f}'.format(grid_search.score(X_test, y_test)))
print('Feature importances:\n{}\n\n'.format(
    grid_search.best_estimator_.named_steps['forest'].feature_importances_),
    sep='')
print()


plt.matshow(grid_search.cv_results_['mean_test_score'].reshape(8, -1),
            vmin=-2.5, vmax=2.5, cmap="RdYlGn", )
plt.xlabel("forest__max_depth")
plt.ylabel("forest__max_features")
plt.xticks(range(len(forest_params['forest__max_depth'])),
           forest_params['forest__max_depth'])
plt.yticks(range(len(forest_params['forest__max_features'])),
           forest_params['forest__max_features'])
plt.colorbar()
plt.title('Shuffled 5-fold')
plt.savefig('/Users/brad/Scripts/python/metis/metisgh/projectluther/cmap2.png')


# Recursive feature selection
# ---------------------------------------------------------------------

select = RFE(LinearRegression(n_jobs=-1), n_features_to_select=1)
select.fit(X_train, y_train)
X_train_rfe = select.transform(X_train)
print('X_train.shape: {}'.format(X_train.shape))
print('X_train_l1.shape: {}'.format(X_train_rfe.shape))

X_test_rfe = select.transform(X_test)
model = LinearRegression(n_jobs=-1).fit(X_train_rfe, y_train)
print('Train score: {:.3f}'.format(model.score(X_train_rfe, y_train)))
print('Test score: {:.3f}'.format(model.score(X_test_rfe, y_test)))
