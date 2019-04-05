import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

df_songs_extra=pd.read_csv('song_extra_info.csv')
df_members=pd.read_csv('members.csv')
df_songs=pd.read_csv('songs.csv')

#将歌曲合并到训练集和测试集，合并标准是歌曲id
df_train=df_train.merge(df_songs,on='song_id',how='left')
df_test=df_test.merge(df_songs,on='song_id',how='left')

#合并用户信息
df_train=df_train.merge(df_members,on='msno',how='left')
df_test=df_test.merge(df_members,on='msno',how='left')

#---------------------------------空值填充-------------------------------
df_train['gender'].fillna(value='Unknown',inplace=True)
df_test['gender'].fillna(value='Unknown',inplace=True)

df_train['source_screen_name'].fillna(value='Unknown',inplace=True)
df_test['source_screen_name'].fillna(value='Unknown',inplace=True)

df_train['source_type'].fillna(value='Unknown',inplace=True)
df_test['source_type'].fillna(value='Unknown',inplace=True)

df_train['genre_ids'].fillna(value='Unknown',inplace=True)
df_test['genre_ids'].fillna(value='Unknown',inplace=True)

df_train['composer'].fillna(value='Unknown',inplace=True)
df_test['composer'].fillna(value='Unknown',inplace=True)

df_train['lyricist'].fillna(value='Unknown',inplace=True)
df_test['lyricist'].fillna(value='Unknown',inplace=True)

#填充歌曲长度的均值
df_train['song_length'].fillna(value=df_train['song_length'].mean(),inplace=True)
df_test['song_length'].fillna(value=df_test['song_length'].mean(),inplace=True)

#语言类型填充出现频率最高的一种
df_train['language'].fillna(value=df_train['language'].mode()[0],inplace=True)
df_test['language'].fillna(value=df_test['language'].mode()[0],inplace=True)


#----------------------------------------填充值处理-------------------------
#genre_ids
df_train['genre_ids']=df_train['genre_ids'].str.split('|')
df_test['genre_ids']=df_test['genre_ids'].str.split('|')
df_train['genre_count']=df_train['genre_ids'].apply(lambda x:len(x) if 'Unknown' not in x else 0)
df_test['genre_count']=df_test['genre_ids'].apply(lambda x:len(x) if 'Unknown' not in x else 0)































