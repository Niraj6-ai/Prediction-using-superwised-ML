#!/usr/bin/env python
# coding: utf-8

# ## THE SPARKS FOUNDATION
#     

# # Predict the percentage of Student based on no.of Study hours

# Step 1: Import Library

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Step 2: Import Data-Set

# In[2]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")


# Step 3: Analysis of Imported Data

# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.info()


# # Step 4: Plot the graph,for Detail Analysis of Data-set 

# In[7]:


plt.scatter(x=data.Hours , y=data.Scores , color='red' , marker='+')
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")                             
plt.show()


# In[8]:



data.plot.bar(x="Hours",y="Scores")


# Step 5: Now we have prepared data for our model

# In[9]:


# data cleaning
data.isnull().sum()


# In[10]:


data.mean()


# Step 6: divide the data into "attributes" (inputs) and "labels" (outputs).

# In[11]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# Step 7: split this data into training and test sets. 

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# test_size is defining how much data we want for testing.so 0.2 means I want to use 20% of data for testing.


# # Select the model and  build the model

# In[13]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 
print("training completed")


# Step 8: Plotting the regression line

# In[14]:


plt.scatter(x=data.Hours , y=data.Scores , color='r' , marker='+')
line = regressor.coef_*X_train+regressor.intercept_
plt.plot(X_train,line,color='b')
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")                             
plt.show()


# Step 9: Making Predictions

# In[15]:


print("shape of X test",X_test.shape)
print(X_test)
print("prediction of score")
y_pred = regressor.predict(X_test)
print(y_pred)


# Step 10: View Actual and Predicted on test set side by side

# In[16]:


df=pd.DataFrame({'Actual': y_test , 'Predicted': y_pred})
df


# In[17]:


# Actual vs Predicted distribution plot
sns.kdeplot(data=y_test, label="Actual",shade=True );
sns.kdeplot(data=y_pred, label ="predicted",shade=True);


# In[18]:


print('train accuracy: ', regressor.score(X_train,y_train),'\ntest accuracy: ', regressor.score(X_test,y_test) )


# Q.What will be the Predicted score if student Studeis for 9.25 hours/day?

# In[19]:


Hours = [[9.25]]
pred = regressor.predict(Hours)
print(pred)


# In[20]:


from sklearn import metrics
print('Mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




