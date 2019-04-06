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

#Artist
print('训练集中歌手数量：',df_train['artist_name'].unique().shape[0])
print('测试集中歌手数量：',df_test['artist_name'].unique().shape[0])
print('训练集和测试集中都出现的歌手数量有：',len(set.intersection(set(df_train['artist_name']),set(df_test['artist_name']))))
df_artists=df_train.loc[:,['artist_name','target']]
artist1=df_artists.groupby(['artist_name'],as_index=False).sum().rename(
    columns={'target':'repeat_count'}
)
artist2=df_artists.groupby(['artist_name'],as_index=False).count().rename(
    columns={'target':'play_count'}
)
#计算歌手出现的比例
df_artist_repeats=artist1.merge(artist2,how='inner',on='artist_name')
print(df_artist_repeats.head())
df_artist_repeats['repeat_percentage']=round(
    (df_artist_repeats['repeat_count']*100)/df_artist_repeats['play_count'],1)
print(df_artist_repeats.head())
df_artist_repeats.drop(['repeat_count','play_count'],axis=1,inplace=True)

#合并到训练集和测试集
df_train=df_train.merge(df_artist_repeats,on='artist_name',how='left').rename(
    columns={'repeat_percentage':'artist_repeat_percentage'})
df_test=df_test.merge(df_artist_repeats,on='artist_name',how='left').rename(
    columns={'repeat_percentage':'artist_repeat_percentage'}
)

#特征处理后，去掉genre_ids，artist_name
df_train.drop(['genre_ids','artist_name'],axis=1,inplace=True)
df_test.drop(['genre_ids','artist_name'],axis=1,inplace=True)
del df_artist_repeats
del df_artists


#composer
df_train['composer']=df_train['composer'].str.split('|')
df_test['composer']=df_test['composer'].str.split('|')

df_train['composer']=df_train['composer'].apply(lambda x:len(x) if 'Unknown' not in x else 0)
df_test['composer']=df_test['composer'].apply(lambda x:len(x) if 'Unknown' not in x else 0)

#source_system_tab
#查看source_system_tab有多少种类型
print('source_system_tab:\n',df_train['source_system_tab'].value_counts())
















