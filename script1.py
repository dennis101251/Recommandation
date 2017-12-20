import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, merge, Reshape, Merge
from keras.layers.merge import Concatenate

#index the user ID
userID_list = pd.read_table('users.txt',header=None,encoding='UTF-8', names = ['user_id']).values
max_user = np.max(userID_list)
userID_hashIndex = np.zeros(max_user + 1)
for i in range(0, len(userID_list)):
    userID_hashIndex[userID_list[i]] = i

trainSet= pd.read_table('netflix_train.txt',delim_whitespace = True,  header=None,encoding='utf-8', names = ['user_id', 'movie_id', 'score', 'time'])
testSet= pd.read_table('netflix_test.txt',delim_whitespace = True, header=None,encoding='utf-8', names = ['user_id', 'movie_id', 'score', 'time'])

users_id_train = trainSet['user_id'].values
movies_id_train = trainSet['movie_id'].values
scores_train = trainSet['score'].values
size_trainSet = len(users_id_train)
user_index_train = np.zeros(size_trainSet)
for i in range(0,size_trainSet):
    user_index_train[i] = userID_hashIndex[users_id_train[i]]
print("load the training data successfully")

users_id_test = testSet['user_id'].values
movies_id_test = testSet['movie_id'].values
scores_test = testSet['score'].values
size_testSet = len(users_id_test)
user_index_test = np.zeros(size_testSet)
for i in range(0,size_testSet):
    user_index_test[i] = userID_hashIndex[users_id_test[i]]
print("load the test data successfully")

n_users = 10000
n_movies = 10000

k = 128
model1 = Sequential()
model1.add(Embedding(n_users + 1, k, input_length = 1 ))
model1.add(Reshape((k,)))
model2 = Sequential()
model2.add(Embedding(n_movies + 1, k, input_length = 1 ))
model2.add(Reshape((k,)))

model = Sequential()
model.add(Merge([model1, model2], mode='dot', dot_axes= 1))
# model.add(Merge([model1, model2], mode='sum'))

model.add(Dropout(0.2))
model.add(Dense(k, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(int(k/4), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(int(k/16), activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1 , activation='linear'))

model.compile(loss = 'mse', optimizer="adam")
print("model is constructed successfully")

X_train = [user_index_train, movies_id_train]
y_train = scores_train
X_test = [user_index_test, movies_id_test]
y_test = scores_test

model.fit(X_train,y_train, validation_data= (X_test, y_test), batch_size = 1000, epochs = 10)





