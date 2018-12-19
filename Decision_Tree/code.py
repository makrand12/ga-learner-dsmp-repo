# --------------
#Importing header files

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split


# Code starts here
data = pd.read_csv(path)
X = data.drop(columns = ["customer.id","paid.back.loan"],axis=1)
print(X.head(5))
y = data["paid.back.loan"]
print(y.head(5))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state = 0)

# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt


#Code starts here

#Storing value counts of target variable in 'fully_paid'
fully_paid=y_train.value_counts()

#Plotting bar plot
plt.bar(fully_paid.index, fully_paid)
plt.show()

#Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
X_train["int.rate"] = X_train["int.rate"].str.replace("%","")
X_test["int.rate"] = X_test["int.rate"].str.replace("%","")
X_train["int.rate"] = X_train["int.rate"].astype(float)
X_test["int.rate"] = X_test["int.rate"].astype(float)
X_train["int.rate"] = X_train["int.rate"] / 100
X_test["int.rate"] = X_test["int.rate"] / 100
num_df = X_train.select_dtypes(include=['number'])
print(num_df.dtypes)
cat_df = X_train.select_dtypes(include = "object")
print(cat_df.dtypes)
# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
cols = list(num_df.columns)
print(cols)
fig, axes = plt.subplots(4,1,figsize =(10,20))
for i in range(0,4):
    sns.boxplot(x=y_train,y=num_df[cols[i]], ax=axes[i])
fig

# Code ends here


# --------------
# Code starts here
cols = list(cat_df.columns)
fig , axes = plt.subplots(2,2,figsize=(8,16))
for i in range(0,2):
    for j in range(0,2):
        sns.countplot(x=X_train[cols[i*2+j]], hue=y_train,ax=axes[i,j])          
        fig.tight_layout() 

# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here

for col in cat_df.columns:
    
    #Filling null values with 'NA'
    X_train[col].fillna('NA',inplace=True)
    
    #Initalising a label encoder object
    le=LabelEncoder()
    
    #Fitting and transforming the column in X_train with 'le'
    X_train[col]=le.fit_transform(X_train[col]) 
    
    #Filling null values with 'NA'
    X_test[col].fillna('NA',inplace=True)
    
    #Fitting the column in X_test with 'le'
    X_test[col]=le.transform(X_test[col]) 

#Initialising 'Decision Tree' model    
model=DecisionTreeClassifier(random_state=0)

#Training the 'Decision Tree' model
model.fit(X_train, y_train)

#Finding the accuracy of 'Decision Tree' model
acc=model.score(X_test, y_test)

#Printing the accuracy
print(acc)





# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model_2 = DecisionTreeClassifier(random_state = 0)
p_tree = GridSearchCV(estimator=model_2,param_grid=parameter_grid,cv=5)
p_tree.fit(X_train,y_train)
acc_2 =p_tree.score(X_test,y_test)
print(acc_2)

# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data = tree.export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None,
                                feature_names=X.columns, filled = True,  
                                class_names=['loan_paid_back_yes','loan_paid_back_no'])

graph_big = pydotplus.graph_from_dot_data(dot_data)
Image(graph_big.create_png())
# Code ends here


