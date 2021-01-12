#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')

# Introduction
This data set is a Beer data-set we are expected to build a Machine Learning model which predicts the overall rating of the beer. (“review/overall” column in “train.csv” is your dependent variable.)
The .csv contains the following columns:
- index - an identifier for the review
- beer/ABV - the alcohol by volume of the beer
- beer/beerId - a unique ID indicating the beer reviewed
- beer/brewerId - a unique ID indicating the brewery
- beer/name - name of the beer
- beer/style
- review/appearance - rating of the beer's appearance (1.0 to 5.0)
- review/aroma - rating of the beer's aroma (1.0 to 5.0)
- review/overall - rating of the beer overall (1.0 to 5.0)
- review/palate - rating of the beer's palate (1.0 to 5.0)
- review/taste - rating of the beer's taste (1.0 to 5.0)
- review/text - the text of the review
- review/timeStruct - a dict specifying when the review was submitted
- review/timeUnix
- user/ageInSeconds - age of the user in seconds
- user/birthdayRaw
- user/birthdayUnix
- user/gender - gender of the user (if specified)
- user/profileName - profile name of the user


# In[5]:


import pandas as pd
import numpy as np


# In[3]:


beer_data=pd.read_csv('/content/drive/MyDrive/Data Sets/beer_data/train.csv')
# beer_data.head()


# # Feature Extraction and Feature Engineering
# We will remove the irrelevant features from the data set by looking at the number of missing data points in them and number of unique values present in the categorical features.
# - Handling the missing data features
# - Categorical Features
# - review/text feature

# In[ ]:


# remove those features which have missing data more than 50% of data points
missing_data=beer_data.isnull().sum()/len(beer_data)*100
import matplotlib.pyplot as plt
plt.figure()
plt.barh(missing_data.index,missing_data.values,height=0.3)
plt.xticks(rotation=90)
plt.title('% of missing_data present in each feature')


# We will remove those columns which have missing values more than 50% of total data points, add these feature in the remove_col list and in the end remove all of them together 

# In[ ]:


remove_col=[col for col in beer_data.columns if beer_data[col].isnull().sum()/len(beer_data)*100 > 50]
#remove_col


# ### Categorical Data
# After removing the irrelevant features from the data set we will move forward to handle the Categorical features and extract the relevant information which make the feature more informative and understandable for the ML model.All the features
# contains specific number classes in it, but most of them already present in integer or float datatype so these are not our concern, We will only look at those feature which are still present in string or character datatype.

# In[ ]:


cat_col=[col for col in beer_data.columns if beer_data[col].dtype=='object']
#beer_data[cat_col].head()


# In[ ]:


#how many unique values each categorical column contains
unique=beer_data[cat_col].nunique()/len(beer_data)*100
#unique


# In[ ]:


plt.figure()
plt.barh(unique.index,unique.values)
plt.xticks(rotation=45)
plt.title("% of unique values present in Categorical features")


# In[ ]:


# The review/timeStruct seems to be a irrelevant feature so we will remove it
remove_col=remove_col+['review/timeStruct']
beer_data.drop(remove_col,axis=1,inplace=True)


# ### review/text feature 
# This feature seems to be a informative, We divide the review/text into 5 different groups these can be considered as excellent, good, neutral, bad, poor and label them by using Topic Modelling.

# In[ ]:


import nltk
beer_data.dropna(inplace=True)
beer_data.reset_index(inplace=True)
text=beer_data['review/text']
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import gensim
from gensim import corpora,models,matutils
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
exclude = set(string.punctuation)
stopword=set(stopwords.words('english'))
def clean_text(doc):
  text1=[i for i in doc.lower().split() if i not in stopword]
  text2=[i for i in text1 if i not in exclude]
  text3=[w for w in text2 if re.search('^\D',w)]
  text4=[nltk.WordNetLemmatizer().lemmatize(i) for i in text3]
  text5=" ".join([w for (w,pt) in nltk.pos_tag(text4) if pt=='JJ' or pt=='NN' or pt=='VBZ'])
  return text5
clean_text=text.apply(lambda x:clean_text(x))
clean_text


# In[2]:


#@title Default title text
dictnry=corpora.Dictionary([t.split() for t in clean_text])
corpus=[dictnry.doc2bow(w.split()) for w in clean_text]
tfidf_cp=models.TfidfModel(corpus,dictionary=dictnry)
lda=gensim.models.ldamodel.LdaModel
LDA=lda(corpus,id2word=dictnry,passes=30,num_topics=2)
LDA.print_topics(num_topics=2,num_words=10)
doc=[w.split() for w in clean_text]
topics=[doc for doc in LDA[corpus]]
beer_data['topics']=pd.Series([max(w,key=lambda x:x[1])[0] for w in topics])


# In[1]:


LDA.print_topics(num_topics=2,num_words=20)


# In[ ]:


beer_data.nunique()


# In[ ]:


# remove some Unwanted columns here
beer_data.drop(['level_0','index','review/text','review/timeUnix','beer/beerId','user/profileName'],inplace=True,axis=1)


# In[ ]:


beer_data.info()
#beer_data.nunique()


# # Exploratory Data Analysis: Graph and Visuals
# We have prepared our dataset completely now we will do some visualization and try extract some insights from the visuals that we going to draw here.
# 

# import seaborn as sns
# plt.figure()
# sns.countplot(hue='review/overall',y='topics',data=beer_data)
# plt.xticks(rotation=90)
# #Most of the comments or text/reviews belong to rating of 4.0 in the datasets.

# In[ ]:





# plt.figure()
# sns.swarmplot(x=beer_data['review/overall'],y=beer_data['comment'])

# In[ ]:


#To draw pairplot we extract feature which have unique value less 100
pp_col=[col for col in beer_data.columns if beer_data[col].nunique()<=10]
df_pp=beer_data[pp_col]
df_pp['review_nature']=beer_data['review/overall']>4.0
#we divided the data points on the basis of two classes in review_nature which gives nature of the overall/review 
#whether the review was positive or negative  


# In[ ]:


sns.pairplot(df_pp.drop('review/overall',axis=1),hue='review_nature')


# By looking at the pairplot we can see that data points are not overlapping and data points are linearly seperable, so we will try to separate the data points by using linear function or Classifier.

# In[ ]:


plt.figure()
plt.barh(beer_data['review/overall'].value_counts().index.astype(str),beer_data['review/overall'].value_counts()/len(beer_data)*100)
# here we can see the Imbalanced data set, 4.0 rating has almost 40% of the sample


# # Model Selection
# - encoding of categorical variables
# - Normalization/Feature Scaling
# - Hyperparameter Tuning

# In[ ]:


from sklearn.preprocessing import LabelEncoder
y=beer_data['review/overall']
X=beer_data.drop('review/overall',axis=1)
cat_col=[col for col in beer_data.columns if beer_data[col].dtype=='object']
print(cat_col)
encoder=LabelEncoder()
for col in cat_col:
    X[col]=encoder.fit_transform(beer_data[col])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


# # New Section

# In[ ]:


column=X.columns
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X)
X_S=pd.DataFrame(scaler.transform(X),columns=column)
X_trainS=pd.DataFrame(scaler.transform(X_train),columns=column)
X_testS=pd.DataFrame(scaler.transform(X_test),columns=column)
X_S.head()


# In[ ]:


# Using Regression model for evaluation
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import cross_validate
model1=LinearRegression()
model2=Ridge()
model3=Lasso()
metric=('r2','neg_mean_squared_error','neg_mean_absolute_error')
cv_score1=pd.Series(cross_validate(model1,X_S,y,scoring=metric,cv=5))
cv_score2=pd.Series(cross_validate(model2,X_S,y,scoring=metric,cv=5))
cv_score3=pd.Series(cross_validate(model3,X_S,y,scoring=metric,cv=5))
Regression_df=pd.DataFrame([cv_score1.apply(lambda x:x.mean()),cv_score2.apply(lambda x:x.mean()),cv_score3.apply(lambda x:x.mean())],index=['Linear Regression','Ridge','Lasso']).drop(['fit_time','score_time'],axis=1)
Regression_df


# In[ ]:


#Using MultiClass Classification Model for evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
model4=LogisticRegression(multi_class='ovr')
model4.fit(X_trainS,y_train.astype(str))
accuracy=accuracy_score(y_test.astype(str),model4.predict(X_testS))
precision=precision_score(y_test.astype(str),model4.predict(X_testS),average='micro')
recall=recall_score(y_test.astype(str),model4.predict(X_testS),average='micro')
f1=f1_score(y_test.astype(str),model4.predict(X_testS),average='micro')
con_matrix=confusion_matrix(y_test.astype(str),model4.predict(X_testS))
#plt.figure()
#sns.heatmap(con_matrix,annot=True)
print('accuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(accuracy,precision,recall,f1))


# #Hyperparameter Tuning
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# model6=SVC(kernel='rbf')
# param_val={'gamma':[0.001,0.01,0.1,1,10],'C':[0.01,0.1,1,10,100]}
# grid=GridSearchCV(model6,scoring='accuracy',param_grid=param_val)
# grid.fit(X_trainS,y_train.astype(str))

# In[ ]:


#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
model5=LogisticRegression(multi_class='ovr')
grid_val={'C':[0.01,0.1,10,50,100,200,500,1000]}
grid=GridSearchCV(model5,param_grid=grid_val,scoring='accuracy')
grid.fit(X_trainS,y_train.astype(str))


# In[ ]:


print(grid.best_params_)
print(grid.best_score_)
accuracy_score(y_test.astype(str),grid.predict(X_testS))


# In[ ]:





# In[ ]:




