#!/usr/bin/env python
# coding: utf-8

# # 1. Business Understanding
# The objectives of this data mining are:
# 
# Knowing the best machine learning algorithms for breast cancer diagnosis classification
# The machine learning algorithms used for comparison are k-NN, Naive Bayes, and Decision Tree
# 2. Data Understanding
# # 2. Data Understanding

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[1]:


data = pd.read_csv("data.csv")


# In[46]:


data.head()


# In[47]:


data.info()


# In[48]:


data.columns


# # 3. Data Preparation
# 3.1. Check Missing Values
# 

# In[49]:


data.isnull().sum()


# # 3.2. Drop Column id

# In[50]:


data.drop('id', axis=1, inplace=True)

data.head()


# # 3.3. Exploratory Data Analysis (EDA)
# 3.3.1. Univariate Analysis

# In[51]:


# Bar Chart - Attribute diagnosis

plt.figure(figsize=(4,4))
sns.countplot(data['diagnosis'])


# In[52]:


# Histogram Bar - Attribute radius_mean
plt.figure(figsize=(4,4))
sns.histplot(data['radius_mean'])


# In[53]:


# Box Plot - Attribute texture_mean
plt.figure(figsize=(4,4))
sns.boxplot(data['texture_mean'])


# # 3.3.2. Bivariate Analysis

# In[54]:


# Scatter Plot - Attribute radius_mean and Attribute perimeter_mean
plt.figure(figsize=(4,4))
sns.scatterplot(x=data['radius_mean'], y=data['perimeter_mean'], hue=data['diagnosis'])


# There is a close positive relationship between attribute radius_mean and perimeter_mean.
# 
# The increase that occurred in attribute radius_mean was also followed by an increase in attribute perimeter_mean. And if attribute radius_mean, attribute perimeter_mean also decreased.

# # 3.3.3. Multivariate Analysis

# In[56]:


# Plotting heatmap based on correlation between attributes
sns.pairplot(data, hue="diagnosis", vars=["radius_mean", "texture_mean", "perimeter_mean", "radius_worst", "perimeter_worst"])
plt.show()


# In[57]:


# Plotting heatmap based on correlation between attributes

plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='YlGnBu')
plt.show()


# # 4. Modeling

# In[58]:


# library to divide the dataset into training data and testing data
from sklearn.model_selection import train_test_split

#library to calculate model performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[59]:


# Separate between attributes (independent, feature) and class (dependent)
x = data.drop(columns='diagnosis')
x.head()


# In[60]:


y = data['diagnosis']
y.head()


# In[61]:


# Separate data into training data and testing data, with a proportion of 70% training data, and 30% testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

print('data training:')
print(xtrain.shape)
print(ytrain.shape)
print('--------------')
print('data testing:')
print(xtest.shape)
print(ytest.shape)


# # 4.1. k-NN Algorithm

# In[62]:


# import library k-NN
from sklearn.neighbors import KNeighborsClassifier


# In[21]:


# using the k-NN algorithm, with the value of k being 5, calculating the euclidean distance (p=2)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2, metric='euclidean')
knn.fit(xtrain, ytrain)
ypred = knn.predict(xtest)


# In[22]:


# calculate accuracy value
print(accuracy_score(ytest, ypred))


# # 4.2. Naive Bayes Algorithm

# In[24]:


# import library Naive Bayes Gaussian
from sklearn.naive_bayes import GaussianNB
# using the Naive Bayes algorithm
nb = GaussianNB()
nb.fit(xtrain,ytrain)
ypred = nb.predict(xtest)
print(accuracy_score(ytest, ypred))


# # 4.3. Decision Tree Algorithm

# In[63]:


from sklearn.tree import DecisionTreeClassifier
# using the Decision Tree algorithm
dt = DecisionTreeClassifier()
dt.fit(xtrain, ytrain)
ypred = dt.predict(xtest)
print(accuracy_score(ytest, ypred))


# # svm

# In[68]:


from sklearn.svm import SVC
svm = SVC(C=1, kernel='rbf', gamma='scale')
svm.fit(xtrain, ytrain)
ypred = svm.predict(xtest)
print(accuracy_score(ytest, ypred))


# In[ ]:




