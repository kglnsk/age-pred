import numpy as np
import pandas as pd 
import sklearn

import catboost as cb
import numpy as np
import pandas as pd

friendships = pd.read_csv('/tmp/data/friends.csv')
ages = pd.read_csv('/tmp/data/test.csv')
edus = pd.read_csv('/tmp/data/testEducationFeatures.csv')
groups = pd.read_csv('/tmp/data/testGroups.csv')

friendships = friendships.set_index('uid')
ages = ages.set_index('uid')
groups = groups.set_index('uid')
edus = edus.set_index('uid')

train = ages.copy()
train['num_friends'] = [len(friendships[friendships.index==uid]) for uid in (train.index)]
train['num_groups'] = [len(groups[groups.index==uid]) for uid in (train.index)]
train['friend_mean_age'] = [ages.loc[ages.index.intersection(friendships[friendships.index==uid].fuid.values),:]['age'].mean() for uid in (train.index)]
train['friend_median_age'] = [ages.loc[ages.index.intersection(friendships[friendships.index==uid].fuid.values),:]['age'].median() for uid in (train.index)]
train['age_from_school'] = [17.5 + 2021 - edus.loc[uid]['school_education'] for uid in (train.index)]
train['higher_education'] = [23.5 + 2021 - edus.loc[uid]['graduation_5'] for uid in (train.index)]
train = train.merge(edus, left_index=True, right_index=True)

X = train.copy()
X = X.iloc[:,1:]

model = cb.CatBoostRegressor().load_model('model')

pred = model.predict(X)
df = pd.DataFrame()
df['uid'] = ages.index
df = df.set_index('uid')
df['age'] = pred

df.to_csv('/var/log/result')
