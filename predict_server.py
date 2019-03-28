#!/usr/bin/env python
# coding: utf-8

# # Using a Machine Learning Modal to assign Change Request
# ### Importing historical work order data from previous year

import pandas as pd               # pandas is a datafram library
import matplotlib.pyplot as plt   # matplotlib.pyplot plots data
import numpy as np                # numpy provides N-dimensional object support
from flask import Flask, jsonify, request

app = Flask(__name__)

df = pd.read_csv("./ChangeHistoryData.csv")   # using a copy of the html file I saved from the URL above

df.shape   # This will show number of rows in the dataframe and number of columns, shows us 6 columns and 387 rows of data

df.isnull().values.any()    # will return True if there are any cells with a null value

del df['ChangeID']  # The ChangeID field is of no real value when making predictions so we will remove it


# ## Preparing the Data and converting to numbers algorithm will understand


import json  #here we import all the map files into dictionary varibles
with open('team_map.json', 'r') as file1:
    team_map = json.load(file1)
    
with open('classification_map.json', 'r') as file2:
    classification_map = json.load(file2)

with open('requestor_map.json', 'r') as file3:
    requestor_map = json.load(file3)

with open('location_map.json', 'r') as file4:
    location_map = json.load(file4)
  
with open('client_map.json', 'r') as file5:  # Using the fixed client map file.
    client_map = json.load(file5)
    
df['Team'] = df['Team'].map(team_map)   # Now use pandas to iterate through each column and map the values to numbers
df['Requestor'] = df['Requestor'].map(requestor_map) 
df['Classification'] = df['Classification'].map(classification_map)
df['Client'] = df['Client'].map(client_map)
df['Location'] = df['Location'].map(location_map)

    
# ## Finally training our Decision Tree Modal

from sklearn.model_selection import train_test_split  # importing from SciKit-Learn
train, test = train_test_split(df, test_size=0.3)  # Split into 70% training data and 30% for testing data

from sklearn.tree import DecisionTreeClassifier  # import Decision Tree Classifier from SKLearn
classifier = DecisionTreeClassifier(max_leaf_nodes=10)  # train the model adjusting number of max leaf nodes, 10 seems best
feature_training_columns = ["Requestor", "Classification", "Client", "Location"] # Here we are using all the columns except Team
classifier = classifier.fit(train[feature_training_columns], train["Team"])  # The Team Column is what we are trying to predict

predictions = classifier.predict(test[feature_training_columns]) # Use the trained modal to make perdictions

from sklearn.metrics import accuracy_score      # Checking the accuracy of our Trained Modal
accuracy_score(test["Team"], predictions)  # Looks like we still get above 80 accuracy most times

# ## Random Forest Tree Algorithm
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,  max_leaf_nodes=20)
clf

def checkAccuracy(clf):
    clf=clf.fit(train[feature_training_columns], train["Team"])
    predictions = clf.predict(test[feature_training_columns])
    return accuracy_score(test["Team"], predictions)

checkAccuracy(clf)

# clf.predict(realdata)
# for item in answers :
#     print(inverted_team[np.asscalar(item)])

# Here we reverse the technical_team_map file to decode the predicted number back to an assignee team and print results
inverted_team = dict([[v,k] for k,v in team_map.items()])

# Using Flash to server the prediction model as an REST API

@app.route('/predict/<int:requestor>/<int:classification>/<int:client>/<int:location>')
def make_prediction(requestor, classification, client, location):
    data = [[requestor, classification, client, location]]
    print(data[0])
    answers = clf.predict(data)

    for item in answers :
        predicted_teamid = np.asscalar(item)
        predicted_team = (inverted_team[np.asscalar(item)])
        print(predicted_team)
    return jsonify({'team': predicted_team, 'teamID' : predicted_teamid})


app.run(host='0.0.0.0', port=5000)
