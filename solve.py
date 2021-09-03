import sklearn
import os 
import catboost as cb
import numpy as np
import pandas as pd

train_folder = '.'
test_folder = '/tmp/data/'


friendships_test = pd.read_csv(os.path.join(test_folder,'friends.csv'))
ages_test = pd.read_csv(os.path.join(test_folder,'test.csv'))
edus_test = pd.read_csv(os.path.join(test_folder,'testEducationFeatures.csv'))
groups_test = pd.read_csv(os.path.join(test_folder,'testGroups.csv'))


friendships_train = pd.read_csv(os.path.join(train_folder,'friends.csv'))
ages_train = pd.read_csv(os.path.join(train_folder,'test.csv'))
edus_train = pd.read_csv(os.path.join(train_folder,'testEducationFeatures.csv'))
groups_train = pd.read_csv(os.path.join(train_folder,'testGroups.csv'))


friendships_test = friendships_test.set_index('uid')
friendships_test['copy_index'] = friendships_test.index
ages_test = ages_test.set_index('uid')
groups_test = groups_test.set_index('uid')
edus_test = edus_test.set_index('uid')


friendships_train = friendships_train.set_index('uid')
friendships_train['copy_index'] = friendships_train.index


ages_train = ages_train.set_index('uid')
groups_train = groups_train.set_index('uid')
edus_train = edus_train.set_index('uid')
groups_train['uid'] = groups_train.index


groups_train = groups_train.merge(ages_train,how = 'left',left_on=groups_train.index, right_on=ages_train.index).set_index('key_0')
age_mean_groups = groups_train.groupby('gid').age.median()
groups_train = groups_train.merge(age_mean_groups,how = 'left', left_on = groups_train.index, right_on = 'gid').set_index('uid')
grouped_age = groups_train.groupby(groups_train.index).age_y.mean()
grouped_age.rename(index={'copy_index':'uid'},inplace=True)

df_test = ages_test.copy()
df_test['num_friends'] = friendships_test.index.value_counts() #[len(friendships_test[friendships_test.index==uid]) for uid in (df_test.index)]
df_test['num_groups'] = groups_test.index.value_counts()
df_test['grouped_age'] = grouped_age    
df_test['friend_mean_age'] = friendships_test.merge(ages_train,how = 'left',left_on='fuid', right_on=ages_train.index).set_index('copy_index').groupby(friendships_test.index).age.mean()
df_test['friend_median_age'] = friendships_test.merge(ages_train,how = 'left',left_on='fuid', right_on=ages_train.index).set_index('copy_index').groupby(friendships_test.index).age.median()
df_test['age_from_school'] = [17.5 + 2021 - edus_test.loc[uid]['school_education'] for uid in (df_test.index)]
df_test['age_from_ptu'] = [18.5 + 2021 - edus_test.loc[uid]['graduation_1'] for uid in (df_test.index)]
df_test['higher_education'] = [22.5 + 2021 - edus_test.loc[uid]['graduation_5'] for uid in (df_test.index)]
df_test['school_education_friends'] = friendships_test.merge(edus_train,how = 'left',left_on='fuid', right_on=edus_train.index).groupby(friendships_test.index).mean()['school_education']
df_test['higher_education_friends1'] = friendships_test.merge(edus_train,how = 'left',left_on='fuid', right_on=edus_train.index).groupby(friendships_test.index).mean()['graduation_1']
df_test['higher_education_friends2'] = friendships_test.merge(edus_train,how = 'left',left_on='fuid', right_on=edus_train.index).groupby(friendships_test.index).mean()['graduation_2']
df_test['higher_education_friends3'] = friendships_test.merge(edus_train,how = 'left',left_on='fuid', right_on=edus_train.index).groupby(friendships_test.index).mean()['graduation_3']
df_test['higher_education_friends4'] = friendships_test.merge(edus_train,how = 'left',left_on='fuid', right_on=edus_train.index).groupby(friendships_test.index).mean()['graduation_4']
df_test['higher_education_friends5'] = friendships_test.merge(edus_train,how = 'left',left_on='fuid', right_on=edus_train.index).groupby(friendships_test.index).mean()['graduation_5']
df_test['higher_education_friends6'] = friendships_test.merge(edus_train,how = 'left',left_on='fuid', right_on=edus_train.index).groupby(friendships_test.index).mean()['graduation_6']
df_test['higher_education_friends7'] = friendships_test.merge(edus_train,how = 'left',left_on='fuid', right_on=edus_train.index).groupby(friendships_test.index).mean()['graduation_7']

df_test = df_test.merge(edus_test, left_index=True, right_index=True)

interest_groups = pd.DataFrame(groups_train['gid'].value_counts())[:3000]
for interest_group in (interest_groups.index):
    group_index = groups_test.loc[groups_test['gid']==interest_group].index
    df_test[str(interest_group)] = 0
    df_test.loc[group_index,str(interest_group)] = 1

X = df_test.copy()
#X = X.drop(['age'] , axis = 'columns')

model = cb.CatBoostRegressor().load_model('model_3000')

pred = (model.predict(X)) #+ xgb_loaded.predict(X.drop(['age','registered_year'],axis='columns')))/2.0
df = pd.DataFrame()
df['uid'] = ages_test.index
df = df.set_index('uid')
df['age'] = pred

df.to_csv('/var/log/result')