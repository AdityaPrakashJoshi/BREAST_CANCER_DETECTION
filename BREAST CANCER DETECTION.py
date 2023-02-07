#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[3]:


#load data
df = pd.read_csv('data.csv')
df.head(7)


# In[4]:


#Count the number of rows and columns in the daha set
df.shape


# In[5]:


#count the number of empty values(NaN, NAN, na) in each column
df.isna().sum()


# In[6]:


#drop the column with all missing values
df = df.dropna(axis=1)


# In[7]:


#get the new count of the number of rows and columns
df.shape


# In[8]:


#get a count ofthe number of Malignant (M) or Benign (B) cells
df.diagnosis.value_counts()


# In[9]:


#visualize the count
sns.countplot(df.diagnosis,label="count")
plt.show()


# In[10]:


#look at the data types to see which columns need to be encoded
df.dtypes


# In[11]:


#encode the categorical data values
from sklearn.preprocessing import LabelEncoder
labelEncoder_Y=LabelEncoder()
df.iloc[:,1]=labelEncoder_Y.fit_transform(df.iloc[:,1].values)


# In[12]:


#create a pair plot
sns.pairplot(df.iloc[:,1:6],hue="diagnosis")
plt.show()


# In[13]:


#print the first 5 rows of the new data
df.head()


# In[14]:


#get the correlation of the columns
df.iloc[:,1:12].corr()


# In[15]:


#visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot=True,fmt=".0%")
plt.show()


# In[16]:


#Split the data set into independent(x) and dependent (y) data sets
x=df.iloc[:,2:31].values
y=df.iloc[:,1].values


# In[17]:


#split the data set into 75% training and 25% testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[18]:


#scale the data(feature scaling)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[19]:


#create a function for the models
def models(x_train,y_train):
  #Logistic Regression Model
  from sklearn.linear_model import LogisticRegression
  log=LogisticRegression(random_state=0)
  log.fit(x_train,y_train)
  
  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
  tree.fit(x_train,y_train)
  
  #Random Forest Classifier
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
  forest.fit(x_train,y_train)

  #Print the models accuracy on the training data
  print("[0]Logistic Regression Training Accuracy:",log.score(x_train,y_train))
  print("[1]Decision Tree Classifier Training Accuracy:",tree.score(x_train,y_train))
  print("[2]Random Forest Classifier Training Accuracy:",forest.score(x_train,y_train))
  
  return log,tree,forest


# In[20]:


#Getting all of the models
model = models(x_train,y_train)


# In[21]:


#test model accuracy on confusion matrix
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
  print("Model ", i)
  cm =confusion_matrix(y_test,model[i].predict(x_test))

  TP=cm[0][0]
  TN=cm[1][1]
  FN=cm[1][0]
  FP=cm[0][1]

  print(cm)
  print("Testing Accuracy = ", (TP+TN) / (TP+TN+FN+FP))
  print()


# In[22]:


#show another way to get metrics of the models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model) ):
  print("Model ",i)
  print( classification_report(y_test,model[i].predict(x_test)))
  print( accuracy_score(y_test,model[i].predict(x_test)))
  print()


# In[23]:


#print the prediction of random forest classifier model
pred=model[2].predict(x_test)
print(pred)
print()
print(y_test)


# In[ ]:




