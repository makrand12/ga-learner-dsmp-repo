# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re

# Code starts here
df = pd.read_csv(path)
print(df.head())
print(df.info())
df['INCOME'] = df['INCOME'].str.replace('$','')
df['HOME_VAL'] = df['HOME_VAL'].str.replace(',','')
df['HOME_VAL'] = df['HOME_VAL'].str.replace('$','')
df['INCOME'] = df['INCOME'].str.replace(',','')
df['BLUEBOOK'] = df['BLUEBOOK'].str.replace('$','')
df['BLUEBOOK'] = df['BLUEBOOK'].str.replace(',','')
df['OLDCLAIM'] = df['OLDCLAIM'].str.replace('$','')
df['OLDCLAIM'] = df['OLDCLAIM'].str.replace(',','')
df['CLM_AMT'] = df['CLM_AMT'].str.replace('$','')
df['CLM_AMT'] = df['CLM_AMT'].str.replace(',','')

X = df.drop('CLAIM_FLAG',1)
y = df['CLAIM_FLAG'].copy()
count = y.value_counts()
print(count)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3 , random_state = 6)


# Code ends here


# --------------
# Code starts here
X_train['INCOME'] = X_train['INCOME'].astype(float)
X_train['HOME_VAL'] = X_train['HOME_VAL'].astype(float)
X_train['BLUEBOOK'] = X_train['BLUEBOOK'].astype(float)
X_train['OLDCLAIM'] = X_train['OLDCLAIM'].astype(float)
X_train['CLM_AMT'] = X_train['CLM_AMT'].astype(float)

X_test['INCOME'] = X_test['INCOME'].astype(float)
X_test['HOME_VAL'] = X_test['HOME_VAL'].astype(float)
X_test['BLUEBOOK'] = X_train['BLUEBOOK'].astype(float)
X_test['OLDCLAIM'] = X_test['OLDCLAIM'].astype(float)
X_test['CLM_AMT'] = X_test['CLM_AMT'].astype(float)

print(X_train.isnull().sum())
print(X_test.isnull().sum())


# Code ends here


# --------------

# drop missing values
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)


y_train=y_train[X_train.index]
y_test=y_test[X_test.index]



# fill missing values with mean
X_train['AGE'].fillna((X_train['AGE'].mean()), inplace=True)
X_test['AGE'].fillna((X_test['AGE'].mean()), inplace=True)

X_train['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)
X_test['CAR_AGE'].fillna((X_test['CAR_AGE'].mean()), inplace=True)



X_train['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)
X_test['INCOME'].fillna((X_test['INCOME'].mean()), inplace=True)



X_train['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)
X_test['HOME_VAL'].fillna((X_test['HOME_VAL'].mean()), inplace=True)

print(X_train.shape,X_test.shape)




# --------------
from sklearn.preprocessing import LabelEncoder
# Code starts here
columns = ['PARENT1', 'MSTATUS', 'GENDER', 'EDUCATION', 'OCCUPATION', 'CAR_USE', 'CAR_TYPE', 'RED_CAR' , 'REVOKED']
for i in columns:
    le = LabelEncoder()
    X_train[i]=le.fit_transform(X_train[i].astype(str))
    X_test[i]=le.transform(X_test[i].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

# Instantiate logistic regression
model = LogisticRegression(random_state = 6)

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the f1 score
score = accuracy_score(y_test, y_pred)
print(score)

# calculate the precision score
precision = precision_score(y_test, y_pred)
print(precision)


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

smote = SMOTE(random_state = 6)
X_train,y_train = smote.fit_sample(X_train, y_train)

scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred =model.predict(X_test)
score = accuracy_score(y_test,y_pred)
# Code ends here


