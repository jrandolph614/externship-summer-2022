#!/usr/bin/env python
# coding: utf-8

# In[255]:


#importing the required packages
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import datetime
import seaborn as sns
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import OneClassSVM
from sklearn.dummy import DummyClassifier
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# In[256]:


tweets = pd.read_csv("EscortDataLabelAll.csv")
list(tweets.columns.values)


# In[257]:


sentiment_counts = tweets.SUSPICIOUS.value_counts()
number_of_tweets = tweets.id.count()
print(sentiment_counts)


# In[258]:


#Checking the distribution of the tweets
Tweet_Values = tweets["SUSPICIOUS"].value_counts()
plt.figure(figsize=(15, 10))
plt.title("The distribution of tweets")
sns.barplot(x = Tweet_Values.index, y = Tweet_Values.values)
plt.xlabel("Tweets Category")
plt.show()


# In[259]:


#balancing dataset
New = pd.DataFrame()
length = len(tweets[tweets["SUSPICIOUS"]==0].index)

i = 0

while i<2:
 new = tweets[tweets["SUSPICIOUS"]==i]
 New = New.append(new.sample(length, random_state=42))
 i = i+1

BalancedData = New

BalancedData.describe()


# In[260]:


#Plotting the severity vs number of accidents again to check if the data is now b
BDTweets = BalancedData["SUSPICIOUS"].value_counts()
plt.figure(figsize=(15, 10))
plt.title("The distribution of tweets")
sns.barplot(x = BDTweets.index, y = BDTweets.values)
plt.xlabel("Tweet Category")
plt.show()


# In[261]:


BalancedData.info()


# In[262]:


#creating the data processing code by utlizing count vectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))
vectorized_data = count_vectorizer.fit_transform(BalancedData.tweet)

#the vectorized data is used to create the indexed_data
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


# In[263]:


#Creating the target label variable 
Label = BalancedData.SUSPICIOUS


# In[264]:


#creating the dataset into train and test by using the train_test_split package. 

data_train, data_test, Label_train, Label_test = train_test_split(indexed_data, Label, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]


# In[265]:


#creating the SVM model with OneVsRestClassifier and fitting the training dataset to the model
OVCmodel = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
fit_model = OVCmodel.fit(data_train, Label_train)


# In[267]:


#predicting the labels of the test dataset using the fitted model
predictions = fit_model.predict(data_test)


# In[266]:


#accuracy score of the test dataset from the fitted model
fit_model.score(data_test, Label_test)


# In[268]:


#printing the classification score metrics and confusion matrix
#this would give precision, recall, f1-score, support

print(classification_report(predictions,targets_test))
print ('\n')
print(confusion_matrix(predictions,targets_test))


# In[269]:


#Producing confusion matrix heat for OneVsRest Classifier

PredictY = fit_model.predict(data_test)
#In order to analyze the performance of the model
Confusion_Matrix = confusion_matrix(y_true=targets_test, y_pred=PredictY)
Actuals = ["ACTUAL SUSPICIOUS","ACTUAL NON-SUSPICIOUS"]
Predicts = ["PREDICTED SUSPICIOUS","PREDICTED NON-SUSPICIOUS"]
Confusion_Matrix = pd.DataFrame(data=Confusion_Matrix,index=Actuals,columns=Predicts)
plt.figure(figsize=(10, 7))
sns.heatmap(Confusion_Matrix, annot=True, fmt="d", cmap=plt.cm.Blues)
plt.title("OneVsRest Classifier Confusion Matrix")
plt.show()


# In[270]:


'''
Creating the Model 0 - baseline Model for OneVsRest Classifier
https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators
created using dummy classifiers
because the dataset is balanced, the baseline model is set to stratified which ge
'''
baseline_model= DummyClassifier(strategy= 'stratified', random_state=42)
baseline_model.fit(data_train, targets_train)
#checking and getting a score for the baseline model
print("The Test score ", baseline_model.score(data_test, targets_test))


# In[232]:


#Multinominal code


# In[286]:


#creating the text processing code
#reference - https://towardsdatascience.com/step-by-step-twitter-sentiment-analysis-in-python-d6f650ade58d

def tweet_processing(tweet):
    
    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    
    return normalization(no_punc_tweet)


# In[287]:


#re-reading the dataset for Multinominal NB model
tweets = pd.read_csv("EscortDataLabelAll.csv")

#applying the tweet processing function
tweets['tweet_list'] = tweets['tweet'].apply(tweet_processing)


# In[288]:


#creating the dataset into train and test by using the train_test_split package.
Data_train, Data_test, label_train, label_test = train_test_split(tweets['tweet'], tweets['SUSPICIOUS'], test_size=0.2)


# In[289]:


len(Data_train)


# In[290]:


#Baseline model 
baseline_model= DummyClassifier(strategy='stratified', random_state=42)
baseline_model.fit(Data_train,label_train)

baselinepredictions = baseline_model.predict(Data_test)
#checking and getting a score for the baseline model
print("The Test score ", baseline_model.score(baselinepredictions,label_test))


# In[291]:


print(len(Data_test))


# In[292]:


#Machine Learning Pipeline
MNBModel = Pipeline([
    ('bow',CountVectorizer(analyzer=tweet_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[293]:


#fitting the model to the training dataset
MNBModel.fit(Data_train,label_train)


# In[296]:


#getting the accuracy score for the model 

predictions = MNBModel.predict(Data_test)
print(accuracy_score(predictions,label_test))


# In[297]:


#printing the classification score metrics and confusion matrix
#this would give precision, recall, f1-score, support

print(len(predictions))
print(classification_report(predictions,label_test))
print ('\n')
print(confusion_matrix(predictions,label_test))


# In[298]:


#Producing confusion matrix heat for MultinomialNB Classifier

PredictY = pipeline.predict(msg_test)
#In order to analyze the performance of the model
Confusion_Matrix = confusion_matrix(y_true=label_test, y_pred=PredictY)
Actuals = ["ACTUAL SUSPICIOUS","ACTUAL NON-SUSPICIOUS"]
Predicts = ["PREDICTED SUSPICIOUS","PREDICTED NON-SUSPICIOUS"]
Confusion_Matrix = pd.DataFrame(data=Confusion_Matrix,index=Actuals,columns=Predicts)
plt.figure(figsize=(10, 7))
sns.heatmap(Confusion_Matrix, annot=True, fmt="d", cmap=plt.cm.Blues)
plt.title("MultinomialNB Classifier Confusion Matrix")
plt.show()


# In[299]:


#New dataset combined with escorting and sugaring
#Codes under this section has several errors and the sugaring data needs to be analyzed further 

escortAndSugar = pd.read_csv("escortAndSugar.csv")
print(escortAndSugar)


# In[300]:


data = escortAndSugar.loc[1:2284, ['tweet','SUSPICIOUS']]


# In[301]:


data['tweet'] = data['tweet'].astype(str).values.tolist()
print(data)


# In[302]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(str)


# In[303]:


data = clean_dataset(data)


# In[304]:


count_vectorizer = CountVectorizer(ngram_range=(1,2))
vectorized_data = count_vectorizer.fit_transform(data.tweet)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


# In[305]:


targets = data.SUSPICIOUS


# In[306]:


from sklearn.model_selection import train_test_split

data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]


# In[307]:


model = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
fit_model = model.fit(data_train, targets_train)


# In[308]:


fit_model.score(data_test, targets_test)


# In[309]:


Unlabeldata = escortAndSugar.loc[2285:4576, ['tweet','SUSPICIOUS']]
Unlabeldata['tweet'] = Unlabeldata['tweet'].astype(str).values.tolist()
print(data)


# In[310]:


Unlabeldata = clean_dataset(Unlabeldata)
print(Unlabeldata)


# In[311]:


predictions = fit_model.predict(Unlabeldata)


# In[312]:


print(predictions)


# In[313]:


Unlabeltweets['textlist'] = Unlabeltweets['text'].astype(str).values.tolist()


# In[314]:


def  data_cleaning(df, tweets):
    df[tweets] = df[tweets].str.lower()
    df[tweets] = df[tweets].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    return df


# In[315]:


Tweets_cleaned = data_cleaning(Unlabeltweets,'textlist')


# In[316]:


print(Tweets_cleaned)


# In[317]:


len(Tweets_cleaned)


# In[318]:


count_vectorizer = CountVectorizer(ngram_range=(1,2))
vectorized_unlabeldata = count_vectorizer.fit_transform(Tweets_cleaned)


# In[319]:


vectorized_unlabeldata.getnnz


# In[320]:


textlist = Unlabeltweets['text'].astype(str).values.tolist()


# In[321]:


textlist = str(textlist)


# In[322]:


vectorized_unlabeldata = count_vectorizer.fit_transform(textlist)


# In[323]:


float_Tweets_cleaned = float(textlist)  


# In[324]:


Unlabelpredictions = fit_model.predict(float_Tweets_cleaned)


# In[325]:


Unlabelpredictions


# In[ ]:





# In[ ]:




