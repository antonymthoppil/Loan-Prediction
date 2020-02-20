#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction Problem 

# In[ ]:


#importing libraries 

import pandas as pd 
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


data = pd.read_csv('C:/Antony Thoppil/Personal/Projects/Loan Prediction/loan-predication/train_u6lujuX_CVtuZ9i.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


#Making copies of the original data set 

df = data.copy()


# In[ ]:


#dropping the variable Loan_ID as it is a unique value and wont add value to the model
df.drop(columns='Loan_ID',inplace=True)


# In[ ]:


df.info()


# In[ ]:


# correlation of numerical variables 

correlation = df.corr()

sn.heatmap(correlation, annot = True)


# In[ ]:


# we can see from the correlation matrix that there is a correlation between the loan amount and applicant income.
# there is also another intersting observation : there is no correlation between the applicant income and the credit history. 


# In[ ]:


# to seperate columns with obj dtype 

obj_cols = [*df.select_dtypes('object').columns]
obj_cols.remove('Loan_Status')


# In[ ]:


# checking the obj_cols list 

obj_cols


# In[ ]:


plt.figure(figsize =(24,18))

for idx,cols in enumerate(obj_cols):
    plt.subplot(3,3,idx+1)
    sn.countplot(cols, data=df, hue='Loan_Status')


# In[ ]:


# Unmarried people having higher proportion of rejection of loan
# Non graduates having more rejection of loan 
# Semi urban has the highest loan approval proportion


# In[ ]:


# to seperate integer and float dtypes columns from the data

num_cols = [*df.select_dtypes(['Int64','Float64']).columns]


# In[ ]:


# checking the list num_cols

num_cols


# In[ ]:


# as loan amount term and credit history are not continuous variables ; it has to be dropped form the list 

num_cols.remove('Loan_Amount_Term')
num_cols.remove('Credit_History')


# In[ ]:


# checking the num cols list again 

num_cols


# In[ ]:


# plotting continous variables 

plt.figure(figsize=(24,18))
count=1

for cols in num_cols:
    plt.subplot(3,2,count)
    sn.boxplot(x='Loan_Status',y=cols, data=df)
    count+=1
    plt.subplot(3,2,count)
    sn.distplot(df.loc[df[cols].notna(),cols])
    count+=1


# In[ ]:


df.describe()


# In[ ]:


df.isna().sum()


# In[ ]:


df.Loan_Status.replace({'Y': 0, 'N': 1},inplace=True)


# In[ ]:


df['Loan_Status'] = df.Loan_Status.astype(int)


# In[ ]:


dummies = pd.get_dummies(df, drop_first = True)


# In[ ]:


SimImp = SimpleImputer()

train = pd.DataFrame(SimImp.fit_transform(dummies),columns=dummies.columns)


# In[ ]:


train.info()


# In[ ]:


train.sample(5)


# In[ ]:


# making the loan amount term column as binary

train['Loan_Amount_360']=np.where(train.Loan_Amount_Term==360,1,0)
sn.countplot(y='Loan_Amount_360',data=train,hue='Loan_Status')

# we can see that with loan amount 360 being 0 more loan applications are rejected.


# In[ ]:


# Dropping the Loan_Amount_Term column as we have added a new feature representing the same

train.drop('Loan_Amount_Term',inplace=True,axis=1)


# In[ ]:


# to check the correlation between all the variables to draw new insights

correlationt = train.corr()
plt.figure(figsize=(24,18))
sn.heatmap(correlationt, annot = True)


# In[ ]:


# we can see that there is a high correlation between credit history and loan status 


# In[ ]:


#Creating train and test set for modelling

X,Y = train.drop('Loan_Status',axis=1),train.Loan_Status
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=123,stratify=Y)


# In[ ]:


## Modelling


# In[ ]:


#Logistic Regression 

logit = LogisticRegressionCV()
logit.fit(X_train,Y_train)

logit_pred = logit.predict(X_test)
print(accuracy_score(Y_test,logit_pred))

confusion_matrix(Y_test,logit_pred)


# In[ ]:


#Random Forest 

ranfor = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_features = 'sqrt')

ranfor.fit(X_train,Y_train)

ranfor_pred = ranfor.predict(X_test)
print(accuracy_score(Y_test,ranfor_pred))

confusion_matrix(Y_test,ranfor_pred)


# In[ ]:


#KNN Neighbours

KNN_clas = KNeighborsClassifier(n_neighbors=8)

KNN_clas.fit(X_train,Y_train)

KNN_pred = KNN_clas.predict(X_test)
print(accuracy_score(Y_test,KNN_pred))

confusion_matrix(Y_test,KNN_pred)


# In[ ]:


#Naive Bayes

gnb = GaussianNB()

gnb.fit(X_train,Y_train)

gnb_pred = gnb.predict(X_test)
print(accuracy_score(Y_test,gnb_pred))

confusion_matrix(Y_test,gnb_pred)


# In[ ]:


# Random Forest gives the best accuracy in terms of percentage 
# Other reasons why Random Forest outperforms other algorithms. 

# the important metrics in this problem statement is approving of bad loans
# random forest has the least false positive (i.e approving bad loans) and the highest true negative(i.e identifying most loans)


# In[ ]:




