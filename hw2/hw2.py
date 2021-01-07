# %%

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
import missingno as msno

from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from LinearRegress import LinearRegress
# pd.set_option('max.columns', 100)
save_img = True  # 是否将代码生成的图片保存
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)

# %%

def f1(x):
    if x > 32:
        return 'very_fat'
    if x > 28:
        return 'fat'
    if x > 24:
        return 'heavy'
    if x > 18.5:
        return 'normal'
    return 'thin'

def f2(x):
    if x > 40:
        return 'old'
    if x > 20:
        return 'young'
    return 'teen'

def pre(df):
    df['bmit'] = df['bmi'].apply(f1)
    df['aget'] = df['age'].apply(f2)
    df = pd.get_dummies(df)
    return df

# %%

train_raw = pd.read_csv('train.csv')
# 读入数据
plt.figure(figsize=(12, 12))
for i, c in enumerate(train_raw.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(train_raw[c], palette=sns.color_palette("Blues_r")[0:5])
    if train_raw[c].dtype == object:
        plt.xticks(rotation=45)
if save_img:
    plt.savefig('img/2_3.png', bbox_inches='tight')
print(train_raw.head())
train_raw = pre(train_raw)
# 二分类属性的0-1转换
# train_raw = pd.get_dummies(train_raw)
# print(train_raw['region'].unique())
print(train_raw.dtypes)

# %%

train_raw.head()

# %%

plt.figure(figsize=(16, 16))
print(train_raw.describe())
sns.heatmap((train_raw).corr(), annot=True, cmap=sns.color_palette("Blues_r"))
if save_img:
    plt.savefig('img/2_1.png', bbox_inches='tight')

# %%

plt.figure(figsize=(4, 4))
sns.distplot(train_raw['charges'])
if save_img:
    plt.savefig('img/2_2.png', bbox_inches='tight')

# %%

# train_raw['charges'] = np.log(1. + train_raw['charges'])
plt.figure(figsize=(16, 24))
for (i, c) in enumerate(train_raw.columns):
    plt.subplot(7, 4, i + 1)
    sns.histplot(train_raw[c])
plt.savefig('img/2_4.png', bbox_inches='tight')
# print(train_raw.describe())

# %%

y = train_raw['charges'].values.reshape(train_raw.shape[0], 1)

train_raw = train_raw.drop(columns=['charges'])
X = train_raw.values
zoomr = 10000
linear = LinearRegress(X / zoomr, y / zoomr)
linear.train(6400)

# %%

t = linear.w * X.mean(axis=0).reshape(X.shape[1], 1) / zoomr
dd = dict()
for i, c in enumerate(train_raw.columns):
    dd[c] = t[i][0]
print(dd)
print(linear.w * X.mean(axis=0).reshape(X.shape[1], 1))
test_raw = pd.read_csv('test_sample.csv')
ans = pd.read_csv('test_sample.csv')
test_raw = pre(test_raw)
test_raw = test_raw.drop(columns=['charges'])
x = test_raw.values
ans['charges'] = linear.predict(x)
# ans['charges'] = np.exp(linear.predict(x)) - 1.
XE = np.column_stack((X, np.ones(X.shape[0])))
XT = np.column_stack((x, np.ones(x.shape[0])))
w = np.linalg.pinv(XE).dot(y)
ans['charges'] = XT.dot(w)
ans.to_csv('submission.csv', index=False)
if True:
    ans['charges'] = np.zeros((x.shape[0], 1))
    divide_data = KFold(n_splits=5).split(X)
    vsc = 0
    for tidx, vidx in divide_data:
        Xt = X[tidx]
        Xv = X[vidx]
        rfr = RandomForestRegressor()
        rfr.fit(Xt, y[tidx])  # 训练数据
        ans['charges'] = ans['charges'] + rfr.predict(x)
        vsc += rfr.score(Xv, y[vidx])
    print('[RF]acc_score:', vsc / 5)
    ans['charges'] = ans['charges'] / 5
    ans.to_csv('submission.csv', index=False)
    print('okay')

# %%

if False:
    model = xgb.XGBRegressor(eta=0.01)
    model.fit(X=train_raw, y=y.flatten())
    xgb.plot_importance(model)
    param_grid = [
        {'n_estimators': [300, 400, 500],
         'max_features': [2, 4, 6, 8],
         'min_child_weight': [i + 1 for i in range(10)],
         'gamma': [i / 100 for i in range(11)],
         'reg_alpha': np.linspace(0, 10, 11),
         'reg_lambda': np.linspace(40, 60, 11),
         'subsample': [0.8, 0.9, 1.],
         'silent': [1],
         'kfold': [5]
         },
    ]

    grid_search = GridSearchCV(xgb.XGBRegressor(eta=0.01), param_grid, cv=5,
                               scoring='neg_mean_squared_error', verbose=False)
    grid_search.fit(X=train_raw, y=y.flatten())

    print(np.mean(np.square(grid_search.predict(train_raw) - y)))
    ypred = grid_search.predict(test_raw)
    ans['charges'] = ypred
    print(ans['charges'].mean())
    ans.to_csv('submission.csv', index=False)

    print('okay')
