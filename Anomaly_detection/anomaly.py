# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:44:32 2021

@author: tarid
"""

#%% Set up working directory
import os

#get the current working directory
print("Current Working Directory " , os.getcwd())

#change the working directory to the intended path
os.chdir("D:/Program/Private_project/Python_private/Anomaly_detection")

#check the newly set working directory
print("Current Working Directory " , os.getcwd())

#%%

#I will be using the NYC taxi passengers dataset that contains 
#the number of taxi passengers from July 2014 to January 2015 at half-hourly intervals.

import pandas as pd
data = pd.read_csv('nyc_taxi.csv')

#column to date/time
data['timestamp'] = pd.to_datetime(data['timestamp'])

data.head()

#%% create moving-averages

data['MA48'] = data['value'].rolling(48).mean()
data['MA336'] = data['value'].rolling(336).mean()
#%%
import plotly.express as px
from plotly.offline import plot #for offline plotly

fig = px.line(data, x="timestamp", y=['value', 'MA48', 'MA336'], title='NYC Taxi Trips', template = 'plotly_dark')
fig.show()

plot(fig)

#%% Data Preparation

#Since algorithms cannot directly consume date or timestamp data, 
#we will extract the features from the timestamp 
#and will drop the actual timestamp column before training models.

# drop moving-average columns
data.drop(['MA48', 'MA336'], axis=1, inplace=True)

# set timestamp to index
data.set_index('timestamp', drop=True, inplace=True)

# resample timeseries to hourly 
data = data.resample('H').sum()

# creature features from date
data['day'] = [i.day for i in data.index]
data['day_name'] = [i.day_name() for i in data.index]
data['day_of_year'] = [i.dayofyear for i in data.index]
data['week_of_year'] = [i.weekofyear for i in data.index]
data['hour'] = [i.hour for i in data.index]
data['is_weekday'] = [i.isoweekday() for i in data.index]

data.head()

#%% Experiment Setup

#Common to all modules in PyCaret, the setup function is the first and the
#only mandatory step to start any machine learning experiment in PyCaret.

#Besides performing some basic processing tasks by default, PyCaret also
#offers a wide array of pre-processing features.

# init setup
from pycaret.anomaly import *
s = setup(data, session_id = 123)

#%% Model training

#Whenever you initialize the setup function in PyCaret, 
#it profiles the dataset and infers the data types for all input features.
#In this case, you can see day_name and is_weekday is inferred as categorical
#and remaining as numeric. You can press enter to continue.

# check list of available models
models()

#In this tutorial, I am using Isolation Forest, 
#but you can replace the ID ‘iforest’ in the code below with any other model ID to change the algorithm.

# train model
iforest = create_model('iforest', fraction = 0.1)
iforest_results = assign_model(iforest)
iforest_results.head()

#Notice that two new columns are appended i.e. Anomaly that contains value 1 for outlier and 0 for inlier 
#and Anomaly_Score which is a continuous value a.k.a as decision function
#(internally, the algorithm calculates the score based on which the anomaly is determined).

# check 10 highest anomalies
iforest_results[iforest_results['Anomaly'] == 1].head(10)

#%% Plot anomaly graph

#We can now plot anomalies on the graph to visualize.

import plotly.graph_objects as go

# plot value on y-axis and date on x-axis
fig_2 = px.line(iforest_results, x=iforest_results.index, y="value", title='NYC TAXI TRIPS - UNSUPERVISED ANOMALY DETECTION', template = 'plotly_dark')

# create list of outlier_dates
outlier_dates = iforest_results[iforest_results['Anomaly'] == 1].index

# obtain y value of anomalies to plot
y_values = [iforest_results.loc[i]['value'] for i in outlier_dates]

fig_2.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))

        
fig_2.show()

plot(fig_2)


#Notice that the model has picked several anomalies around Jan 1st which is a new year eve. 
#The model has also detected a couple of anomalies around Jan 18— Jan 22 which is when the North American blizzard
#(a fast-moving disruptive blizzard) moved through the Northeast dumping 30 cm in areas around the New York City area.

#If you google the dates around the other red points on the graph, 
#you will probably be able to find the leads on why those points were picked up as anomalous by the model (hopefully).

