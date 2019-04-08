import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
pd.set_option('display.width',1000)
pd.set_option('display.max_columns',1000)

df_train=pd.read_csv('df_train.csv')
df_test=pd.read_csv('df_test.csv')
print('read train and test data\n')

y=df_train['target'].values
X=df_train.drop(['msno','song_id','target'],axis=1)


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=1)
print('data has split')
print('X_train head()\n',X_train.head(2))
print('y_train head()\n',y_train)
print('X_val head()\n',X_val.head(2))
print('y_val head()\n',y_val)
#print('val head()\n',val.head())

#X_val.drop(['msno','song_id'],axis=1,inplace=True)

song_ids=df_test['id'].values
X_test=df_test.drop(['msno','song_id','id'],axis=1).values

params={
        'boosting_type':'gbdt',
        'objective':'binary',
        'metric':'auc',
        'learning_rate':0.3,
        'n_jobs':-1
}

lgb_train=lgb.Dataset(X_train,y_train)
lgb_val=lgb.Dataset(X_val,y_val,reference=lgb_train)

model=lgb.train(params=params,train_set=lgb_train,num_boost_round=60,
                valid_sets=lgb_val,
          early_stopping_rounds=5)


y_preds=model.predict(X_test,num_iteration=model.best_iteration)
print(y_preds)
#print('auc is:',model.best_score_)

result_df=pd.DataFrame()
result_df['id']=song_ids
result_df['target']=y_preds

#保存结果
result_df.to_csv('submission.csv',index=False,
                 float_format='%.5f')

