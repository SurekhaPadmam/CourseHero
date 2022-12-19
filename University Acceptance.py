#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
df=pd.read_csv("Book1.csv",skiprows=0)
df=df.drop(['Serial No.'],axis=1)
df.head()


# In[29]:


#look at the columns and their datatypes
df.info()


# In[30]:


#we are gonna check if there are any null values with isnull() method
df.isnull().sum()


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression### Seperating the dataframe for x and y values - Independent and dependent values


# ### Seperating the dataframe for x and y values - Independent and dependent values

# In[32]:



x=df.iloc[:,:-1]
x.head()
y=df.iloc[:,-1]


# #### Now with this, we have x- predictor variables  and y- predictand values

# ### Now, I'm going to see the correlation between x and y values.

# *using whole dataframe*

# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.corr(),annot=True)


# #### Clearly, we can see that CGPA and Chance of Admit has highest correlation value- 0.87. 
# #### The least correlation is with Research and Chance of Admit- 0.55.  so, we are gonna drop this column for better results

# In[34]:


x.head()
x=x.drop(["Research"],axis=1)
x.head()


# ### Splitting Data for our Model

# In[35]:



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) # this 0.2 value splits whole data into 20% test dataset 
# and 80% train dataset
print(x_train.head())
print(y_train.head())
print(y_train.info())


# In[36]:


model=LinearRegression()
model_fit=model.fit(x_train,y_train)


# In[37]:


y_pred=model_fit.predict(x_test.values)


# ### we should now , check the score

# In[38]:


from sklearn.metrics import confusion_matrix,accuracy_score

score=model_fit.score(x_test, y_test)


# In[39]:


from sklearn.metrics import mean_squared_error
# Calculation of Mean Squared Error (MSE)
error=mean_squared_error(y_test,y_pred)
print("Score=",score)
print("Error=",error)


# ### So, Mean Square Error is 0.004 percent
# 
# 
# ## Now, Let's do Principal Component Analysis

# In[40]:


from sklearn.decomposition import PCA
pca=PCA()
data=pca.fit_transform(x)
x.info()
var=pca.explained_variance_ratio_
var=[round(i,3) for i in var]
print(var)


# In[63]:


new_x=x[["GRE Score","CGPA"]]
print(new_x.head())


# ##### Again, splitting the dataset into train and test variables

# In[93]:


newx_train,newx_test,newy_train,newy_test=train_test_split(new_x,y,test_size=0.2)
print(newx_train.head())
print(newy_train.head())


# In[94]:


classifier = LinearRegression()
classifier.fit(newx_train, newy_train)


# In[95]:


newy_pred = classifier.predict(newx_test)


# ## We have predicted new y values. now let's see the score of it

# In[96]:


score2=classifier.score(newx_test, newy_test)
print(score2)


# In[92]:


# I got 83% which is better than the score before


# In[97]:


fig = plt.figure(figsize=(5,7))


sns.relplot(x=new_x["GRE Score"],y=new_x["CGPA"],hue=y,palette="Blues")


# ### From the Plot above, we can see that the chance to getting into University is Higher if GRE and CGPA are high values
# 
# ## Now, new Mean Square Error is

# In[98]:


newerror=mean_squared_error(newy_test,newy_pred)
print(newerror)


# 
# ## We got new error= 0.003 percent which is lower than the previous error
# 
# # So, Clearly, Model that was created using PCA is performing Better than the previous model.
# 
# ### because we got accuracy higher and Mean Square Error lower

# In[101]:


score2=classifier.score(newx_test, newy_test)
print("New Score=",score2*100)
print("New Error=",newerror*100)


# In[ ]:




