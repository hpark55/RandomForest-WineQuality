#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


# data pre-processing
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

# data mining
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix, plot_confusion_matrix
from tqdm import tqdm


# visualization
import matplotlib.pyplot as plt
import matplotlib


# ## Data pre-processing

# In[2]:


wine = pd.read_csv("./winequality-red (1).csv")
wine


# In[3]:


wine.quality = wine.quality.replace(3, 0)
wine.quality = wine.quality.replace(4, 0)
wine.quality = wine.quality.replace(5, 0)
wine.quality = wine.quality.replace(6, 0)
wine.quality = wine.quality.replace(7, 1)
wine.quality = wine.quality.replace(8, 1)
wine.quality = wine.quality.replace(9, 1)
wine.quality = wine.quality.replace(10, 1)


# In[4]:


x = wine.drop(['quality'], axis = 1)
y = pd.DataFrame(wine, columns=['quality'])


# ### 1. Set 70% for training and 30% for testing. Normalize the features.

# In[5]:


#setup
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.3, random_state = 2020)

# normalize features
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.fit_transform(X_test)

print(X_train.shape)
print(X_test.shape)


# ## Learn a model: Random forest

# ### 2. Develop a random forest model when n_estimators = 10 while others are default 
# (1) Train a model  
# (2) Test the model and evaluate it in terms of the Accuracy and confusion matrix
# (3) Discuss the model performance (target:1) in terms of precision and recall

# In[6]:


#(1)Train a model

rfc = RandomForestClassifier(n_estimators = 10, n_jobs = -1, random_state = 2020)
rfc.fit(X_train, y_train.values.ravel()) 
# .values gives an np array, .ravel flattens the data 
# it works simply with y_train


# In[7]:



RandomForestClassifier(n_estimators=10, 
                       criterion='gini', 
                       max_depth=None, 
                       min_samples_split=2, 
                       min_samples_leaf=1, 
                       min_weight_fraction_leaf=0.0, 
                       max_features='auto', 
                       max_leaf_nodes=None, 
                       min_impurity_decrease=0.0, 
                      # min_impurity_split=None, 
                       bootstrap=True, 
                       oob_score=False, 
                       n_jobs=None, 
                       random_state=None, 
                       verbose=0, 
                       warm_start=False, 
                       class_weight=None, 
                       ccp_alpha=0.0, 
                       max_samples=None)


# In[8]:


# train set
y_train_pred = rfc.predict(X_train)
print('Training Accuracy: {:.3f}'.format(accuracy_score(y_train, y_train_pred))) # in terms of accuracy
print(classification_report(y_train_pred, y_train_pred))
print(confusion_matrix(y_train_pred, y_train_pred)) # in terms of confusion matrix
# test set
y_test_pred = rfc.predict(X_test)
print('Testing Accuracy: {:.3f}'.format(accuracy_score(y_test, y_test_pred)))# in terms of accuracy
print(classification_report(y_test_pred, y_test_pred))
print(confusion_matrix(y_test_pred, y_test_pred)) # in terms of consfusion matrix


# 
# (2&3) The accuracy results of the binary classification model on both the training and testing sets are quite high, with 0.992 and 0.89 respectively. However, when dealing with imbalanced data, accuracy may not be the most reliable metric. The target value in this case is the "high quality wine" with a rating of 7 or higher, which is represented by the value 1. In the training set, the precision and recall values for the positive class were both 1, indicating that all positive samples were correctly identified with no false positives or negatives. This suggests that the model performed extremely well in distinguishing between the positive and negative classes.

# In[9]:


help(RandomForestClassifier)


# In[10]:


# (2) Test the model & evaluate algorithm
# train set
y_train_pred = rfc.predict(X_train)
print('Training Accuracy: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))

# test set
y_test_pred = rfc.predict(X_test)
print('Testing Accuracy: {:.3f}'.format(accuracy_score(y_test, y_test_pred)))


# ### 3. Explore a better hyperparameter using GridSearchCV 
# (1) Find out the optimal number of estimators (n_estimators)
# (2) Does the optimal hyperparameter improve the model accuracy? How much is it increased?
# (3) Explain why the accuracy from the GridSearchCV is different from the accuracy from the test set

# In[11]:


rfc = RandomForestClassifier(random_state = 2020)
param_grid = {
    'n_estimators': [47, 50, 60],
    'max_depth': [None,10, 15, 20],
    'max_leaf_nodes': [None, 50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_features':['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 10, verbose = 1, n_jobs = 12)
CV_rfc.fit(X_train, y_train.values.ravel())


# In[12]:


#(1) optimal number of estimators
CV_rfc.best_params_


# The optimal number of estimators are 47.

# In[13]:


help(GridSearchCV)


# In[14]:


result_table = pd.DataFrame(CV_rfc.cv_results_)
result_table = result_table.sort_values(by = 'mean_test_score', ascending = False)
print(result_table[['params','mean_test_score']])


# In[15]:


best_rfc = CV_rfc.best_estimator_
best_rfc.fit(X_train, y_train.values.ravel())

# train set
y_train_pred = best_rfc.predict(X_train)
print('Training Accuracy: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))

# test set
y_test_pred = best_rfc.predict(X_test)
print('Testing Accuracy: {:.3f}'.format(accuracy_score(y_test, y_test_pred)))


# Former testing accuracy was 0.89 and the current testing accuracy is 0.908 which implicates an increase of 0.018

# In[16]:


print(classification_report(y_test, y_test_pred))
plot_confusion_matrix(best_rfc, X_test, y_test)
plt.show()


# In[17]:


help(RandomForestClassifier.feature_importances_)


# (3) The accuracy obtained from GridSearchCV can be different from the accuracy obtained from the test set due to overfitting. GridSearchCV optimizes the model's hyperparameters to fit the validation set, which may not generalize well to new or unseen data. Therefore, it's essential to evaluate the model's performance on an independent test set that is not used during the hyperparameter tuning or training process to obtain a reliable estimate of the model's true performance.

# ### 4. Reporting the final model 
# (1) Plot a confusion matrix for the final model 
# (2) Plot the feature importance of the final model. Discuss which features are important in the classification.
# (3) Think about the most important feature of your final model. Can you show whether the higher value of the feature leads to a higher quality (y = 1) or to a lower quality (y = 0)?

# In[18]:


# (1) confusion matrix
print(classification_report(y_test, y_test_pred))
plot_confusion_matrix(best_rfc, X_test, y_test)
plt.show()


# In[19]:


#(2) plot feature importance
plt.figure(figsize = (6,6))
plt.barh(x.columns, best_rfc.feature_importances_)
plt.xlabel('Importance', fontsize = 20)
plt.ylabel('Features', fontsize = 20)
plt.show()

#alcohol and sulphates have the most important in the classification


# In[20]:


help(RandomForestClassifier.feature_importances_)


# In[21]:


#(3) 
plt.title("Scatter plot of relations of Quality and Alcohol")
x = wine['alcohol']
y = wine['quality']

plt.scatter(x[y==0], y[y==0], color='blue', label='y=0') 
plt.scatter(x[y==1], y[y==1], color='green', label='y=1') 

plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.legend()


# By creating a scatter plot,  binary classification to determine low quality which is 0 and high quality which is 1 on the y axis is conducted , thus we can know whether the high value of feature leads to 1 or low value of feature to 0

# In[ ]:




