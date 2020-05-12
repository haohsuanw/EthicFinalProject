#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('newlytrain.csv')
df2 = pd.read_csv('Testdata.csv')


# In[3]:


train = df[['tokenstitle','tokens','compound','label']]
test = df2[['tokenstitle','tokens','compound']]
train = train.fillna(' ')
test = test.fillna(' ')


# In[4]:


train['total'] =train['tokenstitle']+" "+train['tokens']
test['total'] = test['tokenstitle']+" "+test['tokens']


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


# In[6]:


#tfidf
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(train['total'].values)
tfidf = transformer.fit_transform(counts)


# In[7]:


target = train['label'].values
test_counts = count_vectorizer.transform(test['total'].values)
test_tfidf = transformer.fit_transform(test_counts)


# In[8]:


#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf, target, random_state=0)


# In[9]:


from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)

Extr = ExtraTreesClassifier(n_estimators=5,n_jobs=4)
Extr.fit(X_train, y_train)
print('Accuracy of ExtrTrees classifier on training set: {:.2f}'
     .format(Extr.score(X_train, y_train)))
print('Accuracy of Extratrees classifier on test set: {:.2f}'
     .format(Extr.score(X_test, y_test)))


# In[10]:


from sklearn.tree import DecisionTreeClassifier

Adab= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
Adab.fit(X_train, y_train)
print('Accuracy of Adaboost classifier on training set: {:.2f}'
     .format(Adab.score(X_train, y_train)))
print('Accuracy of Adaboost classifier on test set: {:.2f}'
     .format(Adab.score(X_test, y_test)))


# In[11]:


Rando= RandomForestClassifier(n_estimators=5)

Rando.fit(X_train, y_train)
print('Accuracy of randomforest classifier on training set: {:.2f}'
     .format(Rando.score(X_train, y_train)))
print('Accuracy of randomforest classifier on test set: {:.2f}'
     .format(Rando.score(X_test, y_test)))


# In[12]:


from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(X_train, y_train)
print('Accuracy of NB  classifier on training set: {:.2f}'
     .format(NB.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'
     .format(NB.score(X_test, y_test)))


# In[13]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
print('Accuracy of Lasso classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Lasso classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# In[14]:


extraTree = Extr.score(X_test, y_test)
adaBoost = Adab.score(X_test, y_test)
randomForest = Rando.score(X_test, y_test)
nb = NB.score(X_test, y_test)
logRegression = logreg.score(X_test, y_test)


# In[15]:


value = [[extraTree], [adaBoost], [randomForest], [nb], [logRegression]]
col = ['Extra Tree', 'Ada Boost', 'Random Forest', 'Multinomial NB','Logistic Regression']
accuracy = {
    'Extra Tree': extraTree,
    'Ada Boost':adaBoost,
    'Random Forest':randomForest,
    'Multinomial NB':nb,
    'Logistic Regression':logRegression,
}


# In[16]:


result = pd.DataFrame(data = accuracy, index=["accuracy"])


# In[17]:


result


# In[53]:





# In[54]:


## use Ada classifier
final_result = Adab.predict(test_tfidf)
final = df2.append( pd.DataFrame(final_result), ignore_index = True)
df2['predict_label'] = pd.DataFrame(final_result)
df2.to_csv('labeledada.csv',index=False,encoding='utf-8')


# In[ ]:




