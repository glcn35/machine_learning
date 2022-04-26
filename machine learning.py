#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[8]:


df=pd.read_csv("music.csv")
df


# In[9]:


x=df.drop(columns=["genre"])
x


# In[10]:


y=df["genre"]
y


# In[19]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[14]:


model=DecisionTreeClassifier()
model.fit(x,y)
df


# In[16]:


prediction= model.predict([[21,1],[22,0]])
prediction


# In[18]:


prediction= model.predict([[35,1],[19,0]])
prediction


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[33]:


x=df.drop(columns=["genre"])
y=df["genre"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model.fit(x_train,y_train)
prediction= model.predict(x_test)
score=accuracy_score(y_test,prediction)
score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




