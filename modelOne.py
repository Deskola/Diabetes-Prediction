
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[2]:

dataset = pd.read_csv('diaMod.csv')


# In[4]:

dataset.head(10)


# In[8]:

dataset['BloodPressure']= dataset['BloodPressure'].replace(0,dataset['BloodPressure'].mean())
dataset['BMI']= dataset['BMI'].replace(0.0,dataset['BMI'].mean())


# In[9]:

dataset.head(10)


# In[21]:

X = dataset.iloc[:, :4]
y = dataset.iloc[:,-1]


# In[22]:

#Iplementing model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[23]:

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[24]:

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[25]:

# Saving model to disk
pickle.dump(logreg, open('model.pkl','wb'))


# In[26]:

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9,2, 6]]))


# In[ ]:



