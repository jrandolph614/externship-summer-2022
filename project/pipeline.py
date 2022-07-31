import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = stopwords.words('english')
import re
class pipeline:
    def __init__(self,df):
        self.df = df
        self.process()
    def clean_text(self,df):
        df['clean_text'] = df['tweet'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
        return df
    def tokenize(self,row):
        res = word_tokenize(row['clean_text'])
        return res
    def stop_word_filtering(self,row):
        res = [word for word in row['token'] if word not in stop_words]
        return res
    def lemmatizer(self,row):   
        res = [WordNetLemmatizer().lemmatize(word=word) for word in row['token'] ]
        return res
    def rejoin(self,row):
        res = ''
        for i in row['lement']:
            res+=i
            res+=' '
        return res
    def process(self):
        self.df = self.clean_text(self.df)
        self.df['clean_text'] = self.df['clean_text'].str.lower()
        self.df['token']=self.df.apply(self.tokenize,axis=1)
        self.df['token']=self.df.apply(self.stop_word_filtering,axis=1)
        self.df['lement']=self.df.apply(self.lemmatizer,axis=1)
        self.df['final'] = self.df.apply(self.rejoin,axis=1)
    def returnDf(self):
        return self.df
    def returnX(self,vectorizer):
        self.X = vectorizer.transform(self.df['final'])
        return self.X