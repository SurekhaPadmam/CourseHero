#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# ### Loading CSV file data into data variable

# In[12]:


# First we are reading the Data
data=pd.read_csv("Book2.csv")
print(data.head())


# ### Adding another column "len" which indicates length of each text

# In[13]:


data['len']=data['text'].apply(lambda x: len(x))
print(data.head())


# In[14]:


# to verify the values are correct or not, let's try doing a manual check
print(data["text"][0],"--", len(data["text"][0]))
print(data["text"][1],"--", len(data["text"][1]))
# values are same. so let's continue cleaning the text data


# ### Cleaning the data

# In[15]:


data['text']=data['text'].apply(lambda x:x.strip())
data['len']=data['text'].apply(lambda x: len(x))
print(data.head())


# #### I am assuming, tidying data means, performing data cleaning to the given text. 

# ### So removing punctuation marks, performing tokenization and  removing stopwords

# In[16]:


from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[17]:


print("puctuation marks are : \n",punctuation)
def removepunct(x):
    new=""
    for each in x:
        if each not in punctuation:
            new+= each
    return new
            
data['text']=data['text'].apply(lambda x: removepunct(x))
data['len']=data['text'].apply(lambda x: len(x))
print(data.head())


# ### Now, let's perform tokenization. It splits text/strings into words.

# In[18]:


# see that we have removed .,? in text above, so len values have changed in 2,3,4 rows.
#now,

data['text']=data['text'].apply(lambda x: word_tokenize(x))
data['len']=data['text'].apply(lambda x: len(x))
print(data.head())


# ### Lastly, let's remove frequently occuring stopwords

# In[19]:


stopw=stopwords.words('english')
def removestopword(x):
    l=[]
    for each in x:
        if each in stopw:
            pass
        else:
            l.append(each)
    return l
            

data['text']=data['text'].apply(lambda x: removestopword(x))
data['len']=data['text'].apply(lambda x: len(x))
print(data.head())


# In[20]:


### So, Finally text has been cleaned successfully. let's remove the length column
data=data.drop(['len'],axis=1)
print(data.head())


# In[ ]:




