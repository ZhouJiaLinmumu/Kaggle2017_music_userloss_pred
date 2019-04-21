import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)

df_train=pd.read_csv('train.csv',dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})
df_test=pd.read_csv('test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})

songs_extra=pd.read_csv('song_extra_info.csv')
members=pd.read_csv('members.csv',dtype={'city' : 'category',
                                                      'bd' : np.uint8,
                                                      'gender' : 'category',
                                                      'registered_via' : 'category'})

df_songs=pd.read_csv('songs.csv',dtype={'genre_ids': 'category',
                                                  'language' : 'category',
                                                  'artist_name' : 'category',
                                                  'composer' : 'category',
                                                  'lyricist' : 'category',
                                                  'song_id' : 'category'})

song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = df_train.merge(df_songs[song_cols], on='song_id', how='left')
test = df_test.merge(df_songs[song_cols], on='song_id', how='left')


members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan

songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on = 'song_id', how = 'left')
test = test.merge(songs_extra, on = 'song_id', how = 'left')

#one-hot编码方式
#思考：类别的重要性相同，采用类似one-hot编码方式
gender_train=pd.get_dummies(train['gender'],drop_first=True)
gender_test=pd.get_dummies(test['gender'],drop_first=True)

#拼接
train=pd.concat([train,gender_train],axis=1)
test=pd.concat([test,gender_test],axis=1)

#特征处理后，去掉无用的特征
train.drop(['gender'],axis=1,inplace=True)
test.drop(['gender'],axis=1,inplace=True)

#将年龄分为一个范围,方便转换为类别型特征
train['age_range']=pd.cut(train['bd'],bins=[-45,0,10,18,35,50,80,200])
test['age_range']=pd.cut(test['bd'],bins=[-45,0,10,18,35,50,80,200])

combine=[train,test]
for value in combine:
    value.loc[(value['bd'] > 0) & (value['bd'] <= 10), 'age_category'] = 0
    value.loc[(value['bd'] > 80) & (value['bd'] <= 200), 'age_category'] = 1
    value.loc[(value['bd'] > 50) & (value['bd'] <= 80), 'age_category'] = 2
    value.loc[(value['bd'] > 10) & (value['bd'] <= 18), 'age_category'] = 3
    value.loc[(value['bd'] > 35) & (value['bd'] <= 50), 'age_category'] = 4
    value.loc[(value['bd'] > -45) & (value['bd'] <= 0), 'age_category'] = 5
    value.loc[(value['bd'] > 18) & (value['bd'] <= 35), 'age_category'] = 6


#年龄、年龄范围处理完后，删除不用特征
train.drop(['bd','age_range'],axis=1,inplace=True)
test.drop(['bd','age_range'],axis=1,inplace=True)

'''
#有效的时间
train['validaty_days']=pd.to_timedelta(train['expiration_date']-train['registration_init_time'],unit='d').dt.days
test['validaty_days']=pd.to_timedelta(test['expiration_date']-test['registration_init_time'],unit='d').dt.days
#df_train['validaty_days']=(df_train['expiration_date']-df_train['registration_init_time']).dt.days
#df_test['validaty_days']=(df_test['expiration_date']-df_test['registration_init_time']).dt.days
#处理后，去掉时间特征
train.drop(['expiration_date','registration_init_time'],axis=1,inplace=True)
test.drop(['expiration_date','registration_init_time'],axis=1,inplace=True)'''

del members, df_songs

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

X = train.drop(['target'], axis=1)
y = train['target'].values

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=1)

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

del train, test

import lightgbm as lgb
#d_train = lgb.Dataset(X, y)
#watchlist = [d_train]
lgb_train=lgb.Dataset(X_train,y_train)
lgb_val=lgb.Dataset(X_val,y_val,reference=lgb_train)


#Those parameters are almost out of hat, so feel free to play with them. I can tell
#you, that if you do it right, you will get better results for sure ;)
print('Training LGBM model...')
params={
        'boosting':'gbdt',
        'objective':'binary',
        'metric':'auc',
        'learning_rate':0.2,
        'num_leaves':256,
        'max_depth':10,
        'num_rounds':200,
        'begging_freq':1,
        'begging_seed':1,
        'max_bin':256,
        'n_jobs':-1
}
model=lgb.train(params=params,
                train_set=lgb_train,
                valid_sets=lgb_val,
                early_stopping_rounds=5)

'''
params = {}
params['learning_rate'] = 0.2
params['application'] = 'binary'
params['max_depth'] = 8
params['num_leaves'] = 2**8
params['verbosity'] = 0
params['metric'] = 'auc'
params['n_jobs']=-1

model = lgb.train(params, train_set=d_train, num_boost_round=50, valid_sets=watchlist, \
verbose_eval=5)'''

print('Making predictions and saving them...')
p_test = model.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')


