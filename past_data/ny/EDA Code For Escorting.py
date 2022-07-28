#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


# Importing the datasets

data = pd.read_csv('EscortDataLabelAll.csv')
data.head()


# In[3]:


#checking the number of rows and colums in the data
print("Data has %d rows and %d columns" % data.shape)
data.columns


# In[4]:


#function for unique values in the data
def return_unique_values(data_frame):
    unique_dataframe = pd.DataFrame()
    unique_dataframe['Features'] = data_frame.columns
    uniques = []
    for col in data_frame.columns:
        u = data_frame[col].nunique()
        uniques.append(u)
    unique_dataframe['Uniques'] = uniques
    return unique_dataframe


# In[5]:


unique_values = return_unique_values(data)
print(unique_values)


# In[6]:


#bar plot for unique values in each column
f, ax = plt.subplots(1,1, figsize=(10,5))

sns.barplot(x=unique_values['Features'], y=unique_values['Uniques'], alpha=0.7)
plt.title('Bar plot for Unique Values in Each Column')
plt.ylabel('Unique values', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xticks(rotation=90)
plt.show()


# In[7]:


#function for plotting the frequency of sepcific attributes 

def plot_frequency_charts(df, feature, title, pallete):
    freq_df = pd.DataFrame()
    freq_df[feature] = df[feature]
    
    f, ax = plt.subplots(1,1, figsize=(16,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette=pallete)
    g.set_title("Number and percentage of {}".format(title))

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 

    plt.title('Frequency of {} tweeting in this dataset'.format(feature))
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel(title, fontsize=12)
    plt.xticks(rotation=90)
    plt.show()


# In[8]:


#frequency plot for the number of tweets tweeted by the same author ID
plot_frequency_charts(data, 'author id', 'Author ID','winter')


# In[9]:


#Most frequent search tags in the escorting dataset
plot_frequency_charts(data, 'search_tag', 'Search Tag', 'tab20')


# In[10]:


#Most frequency sources from which the tweets were posted in the dataset
plot_frequency_charts(data, 'source','Source', 'ocean')


# In[11]:


X = data

#Checking the correlation between features
corr_matrix = X.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="seismic")
plt.gca().patch.set(hatch="X", edgecolor="#666")
plt.show()

#interesting, author ID seems to have a correlation


# In[12]:


#We want to drop off the shaded columns that don't correlate at all in the matrix
y = data


# In[15]:


#he columns "location_ind", "Unnamed: 17" are dropped off the data
Drop = ["location_ind", "Unnamed: 17"]
y = y.drop(Drop, axis=1)
y.head()


# In[16]:


#Correlattion for features in the dataset
plt.figure(figsize=(15, 10))
sns.heatmap(y.corr(), cmap="coolwarm", annot = True)
plt.title("Correlation for features")
plt.show()


# In[17]:


X.info()

