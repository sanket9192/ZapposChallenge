# -*- coding: utf-8 -*-
'''
Created on Thu Dec 22 17:39:47 2016
@author: Sanket Patel
Python 2.7

Main Objective: Recommand the top 5 product to the user among the list of available products.

It is not poosible for a user to visit each product of the website. But we can predict according to users interest.
If user visits the some products, we can recoomand the other product by predicting the likelyhood product or 
the product liked by the other user who is having the same mentality.

Two type of approches to measure the similarities between user.
1. User based collaborative filtering
2. Item Based collaborative filtering.

we can recommand products using both the methods. Results will be different but both are having their benifits.  
'''
def predict(visits, similarity, type='user'):
    if type == 'user':
        mean_user_visits = visits.mean(axis=1)
        visits_diff = (visits - mean_user_visits[:, np.newaxis]) 
        pred = mean_user_visits[:, np.newaxis] + similarity.dot(visits_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = visits.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))    

import numpy as np
import pandas as pd
header  = ['user_id', 'item_id', 'visits', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of visits = ' + str(n_items)  

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#find the pairwise distance 
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

#predict the visits of a user on each product 
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

Top_5_of_user_2 = sorted(range(len(user_prediction)), key=lambda i: user_prediction[2][i])[-5:]
Top_5_of_user_10 = sorted(range(len(user_prediction)), key=lambda i: user_prediction[10][i])[-5:]

#showing the recommanded products
print 'Items recommanded to 2nd User  : '+ str(Top_5_of_user_2)
print 'Items recommanded to 10th User : '+ str(Top_5_of_user_10)

#checking the error in our prediction
print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))