import numpy as np
import pandas as pd
import lightgbm as lgbm
import datetime
import math
import gc
#------------------------------加载数据---------------------------------
dtype={'msno':'category',
       'song_id':'category',
       'source_system_tab':'category',
       'source_screen_name':'category',
       'source_type':'category',
       'target':np.uint8
       }
train=pd.read_csv('train.csv',dtype=dtype,encoding='gbk')
dtype={'msno':'category',
       'song_id':'category',
       'source_system_tab':'category',
       'source_screen_name':'category',
       'source_type':'category'
}

test=pd.read_csv('test.csv',dtype=dtype)
dtype={'genre_ids':'category',
       'language':'category',
       'artist_name':'category',
       'composer':'category',
       'lyricist':'category',
        'song_id':'category'
}
songs=pd.read_csv('songs.csv',dtype=dtype)
dtype={'city':'category',
       'bd':np.uint8,
       'gender':'category',
        'registered_via':'category'
}
members=pd.read_csv('members.csv',dtype=dtype)

songs_extra=pd.read_csv('song_extra_info.csv')
print('Data is preprocessing...')


#-------------------------特征工程部分---------------------------------------
#将songs合并到训练集、测试集

train=train.merge(songs,how='left',on='song_id')
test=test.merge(songs,how='left',on='song_id')

#提取出年 月 日
members['registration_year']=members['registration_init_time'].apply(
    lambda x:int(str(x)[0:4])
)
members['registration_month']=members['registration_init_time'].apply(
       lambda x:int(str(x)[4:6])
)
members['registration_day']=members['registration_init_time'].apply(
       lambda x:int(str(x)[6:8])
)
#去掉注册日期
members=members.drop(['registration_init_time'],axis=1)

#提取出音像制品编码中的年份
def isrc_to_year(isrc):
       if type(isrc)==str:
              if int(isrc[5:7])>17:
                     return 1900+int(isrc[5:7])
              else:
                     return 2000+int(isrc[5:7])
       else:
              return np.nan
songs_extra['song_year']=songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc','name'],axis=1,inplace=True)

#将member表格合并到训练集与测试集
train=train.merge(members,how='left',on='msno')
test=test.merge(members,on='msno',how='left')

#将歌曲的额外信息合并到训练集与测试集
train=train.merge(songs_extra,how='left',on='song_id')
test=test.merge(songs_extra,how='left',on='song_id')

#对于歌曲长度缺失值，将其设置为200000ms，
train.song_length.fillna(200000,inplace=True)
train.song_length=train.song_length.astype(np.uint32)
test.song_length.fillna(200000,inplace=True)
test.song_length=test.song_length.astype(np.uint32)

#定义songs列表中，歌曲类别
def genre_id_count(x):
       if x=='no_genre_id':
              return 0
       else:
              return x.count('|')+1 #统计词作者的个数
train['genre_ids']=train['genre_ids'].cat.add_categories['no_genre_id']
train['genre_ids'].fillna('no_genre_id',inplace=True)
test['genre_ids']=test['genre_ids'].cat.add_categories['no_genre_id']
test['genre_ids'].fillna('no_genre_id',inplace=True)

#将没有歌曲类别属性的位置填充0
train['genre_ids_count']=train['genre_ids'].apply(genre_id_count).astype(np.int8)
test['genre_ids_count']=test['genre_ids'].apply(genre_id_count).astype(np.int8)

#词作者
def lyricist_count(x):
       if x=='no_lyricist':
              return 0
       else:
              return sum(map(x.count,['|','/','\\',';']))
#将词作者一列的缺失值，填充为no_lyricist
train['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricist'].fillna('no_lyricist',inplace=True)

#生成词作者数量一列特征
train['lyricist_count']=train['lyricist'].apply(lyricist_count).astype(np.uint8)
test['lyricist_count']=test['lyricist'].apply(lyricist_count).astype(np.uint8)

#作曲
def composer_count(x):
       if x=='no_composer':
              return 0
       else:
              return sum(map(x.count,['|','/','\\',';']))+1
#将作曲composer列缺失值填充为0
train['composer'].fillna('no_composer',inplace=True)
test['composer'].fillna('no_composer',inplace=True)

train['composer_count']=train['composer'].apply(composer_count).astype(
       np.int8
)
test['composer_count']=test['composer'].apply(composer_count).astype(np.int8)

#歌手 artist_name
def is_featured(x):
       if 'feat' in str(x):
              return 1
       else:
              return 0
train['atrist_name']=train['atrist_name'].cat.add_categories(['no_artist'])
train['artist_name'].fillna('no_artist',inplace=True)
train['is_featured']=train['artist_name'].apply(is_featured).astype(np.int8)

test['atrist_name']=test['atrist_name'].cat.add_categories(['no_artist'])
test['artist_name'].fillna('no_artist',inplace=True)
test['is_featured']=test['artist_name'].apply(is_featured).astype(np.int8)


def artist_count(x):
       if x == 'no_artist':
              return 0
       else:
              return x.count('and') + x.count(',') + x.count('feat') + x.count('&')


train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)

# if artist is same as composer
train['artist_composer'] = (train['artist_name'] == train['composer']).astype(np.int8)
test['artist_composer'] = (test['artist_name'] == test['composer']).astype(np.int8)

# if artist, lyricist and composer are all three same
train['artist_composer_lyricist'] = (
               (train['artist_name'] == train['composer']) & (train['artist_name'] == train['lyricist']) & (
                      train['composer'] == train['lyricist'])).astype(np.int8)
test['artist_composer_lyricist'] = (
               (test['artist_name'] == test['composer']) & (test['artist_name'] == test['lyricist']) & (
                      test['composer'] == test['lyricist'])).astype(np.int8)


# is song language 17 or 45.
def song_lang_boolean(x):
       if '17.0' in str(x) or '45.0' in str(x):
              return 1
       return 0


train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)

_mean_song_length = np.mean(train['song_length'])


def smaller_song(x):
       if x < _mean_song_length:
              return 1
       return 0


train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int8)
test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)

# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}


def count_song_played(x):
       try:
              return _dict_count_song_played_train[x]
       except KeyError:
              try:
                     return _dict_count_song_played_test[x]
              except KeyError:
                     return 0


train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)

# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}


def count_artist_played(x):
       try:
              return _dict_count_artist_played_train[x]
       except KeyError:
              try:
                     return _dict_count_artist_played_test[x]
              except KeyError:
                     return 0


train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)



print("Train test and validation sets")
for col in train.columns:
       if train[col].dtype == object:
              train[col] = train[col].astype('category')
              test[col] = test[col].astype('category')

X_train = train.drop(['target'], axis=1)
y_train = train['target'].values

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

# del train, test; gc.collect();

d_train_final = lgbm.Dataset(X_train, y_train)
watchlist_final = lgbm.Dataset(X_train, y_train)























