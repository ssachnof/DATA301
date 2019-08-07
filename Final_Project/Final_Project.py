
# coding: utf-8

# In[1]:


#Juan J Salazar
#Samuel Sachnoff
#Michael Silin 
#DATA301
#Final Project
#Predicting NFL Arrests


import numpy as np
import pandas as pd
from numpy import genfromtxt
df= pd.read_csv('arrests.csv')
df= df.fillna('False')
def getValues(df):
    if df=='OT':
        return 0
    else:
        return 1
df['OT_flag']= df['OT_flag'].transform(getValues)
dfOriginal = df.copy()
df.head()


# In[2]:


def getTeams(df):
    teams= []
    for team in df['home_team']:
        if team not in teams:
            teams.append(team)
    for team in df['away_team']:
        if team not in teams:
            teams.append(team)
    teams= np.array(teams)
    teams= np.sort(teams)
    return teams
teams= getTeams(df)
weekDays= np.array(['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
def vectorizeTeams(team):
    for i in range(len(teams)):
        if teams[i]==team:
            return i
    return -1
def vectorizeWeekDay(weekDay):
    for i in range(len(weekDays)):
        if weekDays[i]==weekDay:
            return i
    return -1
df['home_team']= df['home_team'].transform(vectorizeTeams)
df['away_team']=df['away_team'].transform(vectorizeTeams)
df['day_of_week']=df['day_of_week'].transform(vectorizeWeekDay)
df.head()


# In[3]:


from datetime import datetime
from dateutil import parser
def castGametimes(df):
    dates= []
    for date in df['gametime_local']:
        hour= parser.parse(date)
        #print(date.strftime('%H'))
        #->for now just get the date as is, if want to increase accuracy, round to closest hour or half hour
        dates.append(date)
    df['gametime_local']= np.array(dates)
    return df
def getGametimes(df):
    gametimes= []
    for gametime in df['gametime_local']:
        if gametime not in gametimes:
            gametimes.append(gametime)
    return gametimes
df= castGametimes(df)
gametimes= getGametimes(df)
def vectorizeGametimes(gametime):
    for i in range(len(gametimes)):
        if gametimes[i]==gametime:
            return i
    return -1
def vectorizeIsDivGame(isDivGame):
    if isDivGame == 'n':
        return 0
    else:
        return 1
df['division_game']= df['division_game'].transform(vectorizeIsDivGame)
df['gametime_local']=df['gametime_local'].transform(vectorizeGametimes)
data= df.to_dict('records')
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse= False,dtype =int)
vectorizedData= vec.fit_transform(data)
vectorizedData


# In[4]:


df= pd.DataFrame(vectorizedData, columns= vec.get_feature_names())
df= df.drop(columns= ['arrests=False'])
#use random indices instead
indices= np.arange(0,1006)
np.random.shuffle(indices)
trainingData= df.iloc[indices[:805]]
testingData= df.iloc[indices[805:]]


# In[5]:


#Cross validation

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
pd.options.mode.chained_assignment = None 

def crossValidate(df1):
    errors= [] #Errors  from each iteration of the cross validation
    #get the different x columns to take correlations of
    x_columns= np.array([])
    x_columns= np.append(df1.columns[0],x_columns)
    x_columns= np.append(df1.columns[2:],x_columns)
    y_data= df1['arrests']
    x_data= df1[x_columns]
    kf = KFold(n_splits=10,shuffle= True)
    sampleIndices= [5,12]
    model= make_pipeline(PolynomialFeatures(5),linear_model.LinearRegression())
    for train_index, test_index in kf.split(x_data):
        train_x= x_data.iloc[train_index]
        test_x= x_data.iloc[test_index]
        train_y= y_data.iloc[train_index]
        test_y= y_data.iloc[test_index]
        #normalize the data
        for col in x_columns:
            if train_x[col].corr(train_y)<0:
                train_x[col]=train_x.loc[:,col]*-1
                test_x[col]= test_x.loc[:,col]*-1
        train_x= (train_x-train_x.mean())/train_x.std()
        test_x= (test_x-test_x.mean())/test_x.std()
        model.fit(train_x,train_y)
        #get the error
        y_fit= model.predict(test_x)
        error= (test_y-y_fit).mean()
        errors.append(error)
    errors= np.array(errors)
    print(errors.mean())
    
    #Plot graph of errors
    erroSeries = pd.Series(errors, index=np.arange(1,len(errors)+1))
    for i in range(len(erroSeries)):
        if(erroSeries.iloc[i] < 0):
            erroSeries.iloc[i] *= -1
    erroSeries.plot()

    import matplotlib.pyplot as plt
    plt.ylabel("Percentage Error")
    plt.xlabel("Cross Validation Iteration")
    plt.title("Error At Each Cross Validation Iteration")
    plt.savefig("crossVal.png")
    
    return model


# In[6]:


#predict the test data

model= crossValidate(trainingData)
x_columns= np.array([])
x_columns= np.append(trainingData.columns[0],x_columns)
x_columns= np.append(trainingData.columns[2:],x_columns)
test_x= testingData[x_columns]
test_y= testingData['arrests']
#mult the col by -1 if there's a negative correlation
for col in x_columns:
    if test_x[col].corr(test_y)<0:
        test_x[col]=test_x.loc[:,col]*-1
#normalize the test data
test_x= (test_x-test_x.mean())/test_x.std()
#predict the test data
predictions= model.predict(test_x)

print((predictions-test_y).mean())


# In[7]:


#
#pd.DataFrame([predictions,test_y]).T.plot()


# In[8]:


#Average arrests at each stadium 

def getArrests(df):
    if df=='False':
        return 0
    else:
        return df
dfOriginal['arrests']= dfOriginal['arrests'].transform(getArrests)

numberOfArrests = dfOriginal.groupby(['home_team'])['arrests'].mean()
numberOfArrests.sort_values(ascending=False, inplace=True)
numberOfArrests.plot.bar(figsize=(10,10))

import matplotlib.pyplot as plt2
plt2.ylabel("Average Number of Arrests",fontsize=18)
plt2.xlabel("Home Team",fontsize=18)
plt2.title("Average Arrests in 40 Games",fontsize=18)
#plt2.savefig("averageArrests.png")


# In[9]:


#Total arrests at each stadium 

def getArrests(df):
    if df=='False':
        return 0
    else:
        return df
dfOriginal['arrests']= dfOriginal['arrests'].transform(getArrests)

numberOfArrests = dfOriginal.groupby(['home_team'])['arrests'].sum()
numberOfArrests.sort_values(ascending=False, inplace=True)
numberOfArrests.plot.bar(figsize=(10,10))

import matplotlib.pyplot as plt5
plt5.ylabel("Total Number of Arrests",fontsize=18)
plt5.xlabel("Home Team",fontsize=18)
plt5.title("Total Arrests in 40 Games",fontsize=18)
#plt5.savefig("TotalArrests")


# In[10]:


#Prediction errors after running the program 10 different times
#Got the absolute value of the error

predictionErrors = [0.4430044664340283,6.629911570234311,4.780705840550462,5.522846458282644,2.251654031120297,
                   2.1595179090221746,0.789374753147548,3.620399230834126,3.1870189150780837,6.0439596145066155]

print("Mean predicted error: ",np.array(predictionErrors).mean())
predErrorsDf = pd.DataFrame(predictionErrors,index=np.arange(1,len(predictionErrors)+1))
predErrorsDf.columns=(["Error"])
predErrorsDf

predErrorsDf.plot(figsize=(8,8))
import matplotlib.pyplot as plt3
plt3.ylabel("Prediction Error",fontsize=18)
plt3.xlabel("Running Test",fontsize=18)
plt3.title("Prediction Error of Testing Set",fontsize=18)


# In[11]:


#Prediction error vs polynomial degree

polyDegreeError = [2089068075295.3418,242732702995.5152,0.10274515498496888,2.856690458408637,
                  5.791500538416036,2.3206740396426366,2.9819719228491044,-240.58275213780382]
pdErrorsDf = pd.DataFrame(polyDegreeError,index=np.arange(3,len(polyDegreeError)+3))
pdErrorsDf.columns=(["Error"])

pdErrorsDf.plot(figsize=(8,8))
import matplotlib.pyplot as plt4
plt4.ylabel("Prediction Error",fontsize=18)
plt4.xlabel("Polynomial Degree",fontsize=18)
plt4.title("Prediction Error vs Polynomial Degree",fontsize=18)


# In[12]:


#Arrests grouped by Gametime 

df2 = dfOriginal
def getValues2(df2):
        return int(parser.parse(df2).strftime('%H')) - 12
    
df2['gametime_local']= df2['gametime_local'].apply(getValues2)
numberOfArrests = df2.groupby(['gametime_local'])['arrests'].mean()
numberOfArrests.sort_values(ascending=False, inplace=True)
numberOfArrests.plot.bar(figsize=(10,10))

plt2.ylabel("Average Number of Arrests",fontsize=18)
plt2.xlabel("Game Time",fontsize=18)
plt2.title("Average Arrests in 40 Games",fontsize=18)


# In[13]:


#Arrests grouped by Overtime flag

numberOfArrests = df.groupby(['OT_flag'])['arrests'].sum()
numberOfArrests.sort_values(ascending=False, inplace=True)
numberOfArrests.plot.bar(figsize=(10,10))

plt2.ylabel("Number of Arrests",fontsize=18)
plt2.xlabel("Overtime?",fontsize=18)
plt2.title("Total Arrests in 40 Games",fontsize=18)

