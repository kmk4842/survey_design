#!/usr/bin/env python
# coding: utf-8

# # Survey example statistics
# Workshop for doctoral students and young faculty, based on survey on accessibility conducted at PŁ. It's based on progressive survey questions:
# 1. Asks for overall rating of accesibility using a numbered scale.
# 2. Asks for a rating on 7 elements using categories as scale.
# 3. Asks for a rating on 12 elements of accessible bathroom design (based on technical requirements for accessible bathrooms).

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# pd.set_option('display.float_format', '{:.4f}'.format)


# In[2]:


data = pd.read_csv('./data.csv')
data.columns


# In[3]:


data.head()


# In[5]:


# Creating a new dataframe to use values for statistics instead of labels
df = data.copy()
df.columns=['Time', 'Nick','Overall', 'R_Parking', 'R_Signs', 'R_Bathrooms', 
              'R_Classrooms', 'R_Furniture', 'R_Building', 'R_Floor', 
             'B_Floor', 'B_Find', 'B_Mark', 'B_DoorWide', 'B_DoorEasy', 'B_Toilet', 'B_Paper', 'B_BarsToilet', 
            'B_BarsSink', 'B_Miror', 'B_Sink', 'B_Towel']


# In[6]:


# Change categories to ordered rankings
x = df['R_Parking'].astype('category')
dict(enumerate(x.cat.categories ))
dict_R_ratings = {'Accessible to people with minor requirements' : 2,
 'Accessible to people with some requirements' : 3,
 'Accessible to people with various requirements': 4,
 'Hardly accessible' : 1,
 'Not accessible' : 0}
for c in df.iloc[:,3:10]:
    print('Changing values in column ' + c)
    df[c].replace(dict_R_ratings, inplace = True)
df.iloc[1:5,3:10]


# In[7]:


pd.crosstab(data.iloc[:,3], df.iloc[:,3])


# In[8]:


# Change categories to ordered rankings
x = df['B_Floor'].astype('category')
dict(enumerate(x.cat.categories ))
dict_B_ratings = {'All of the bathrooms': 4,
 "Don't know" : np.nan,
 'Half of the bathrooms' : 2,
 'Hardly ever' : 0,
 'In some cases' : 1,
 'Most of the bathrooms' : 3}
for c in df.iloc[:,10:22]:
    print('Changing values in column ' + c)
    df[c].replace(dict_B_ratings, inplace = True)
df.iloc[1:5,10:22]


# In[9]:


pd.crosstab(data.iloc[:,10], df.iloc[:,10])


# ## Does Q2 form one factor or many?
# * With 2 factors we explain 75% variance, with one only 54%.
# * Only two items (building and floor access) correlate with answers to question 1.
# * The first factor actually loads negatively.
# * This means people did not consider the remaining dimensions of accesibility

# In[40]:


pd.set_option('display.float_format', '{:.4f}'.format)


# In[12]:


from sklearn.decomposition import PCA
x = df.iloc[:,3:10]
pca_model = PCA(n_components=2) #  75% of variance with just two components, 87% with three.
pca_model.fit(x)
print(pca_model.explained_variance_ratio_)
print("Total variance explained is " + str(pca_model.explained_variance_ratio_.sum()))
print(pca_model.explained_variance_ratio_)


# In[13]:


# Loadings
pd.DataFrame(pca_model.components_.transpose(), index = x.columns).sort_values(0)


# In[14]:


# Calculate PCA values per observations
df_pca = pd.DataFrame(pca_model.components_.transpose(), index = x.columns)
result = []
for i in range(2):
    result.append(x.dot(df_pca.iloc[:,i]))
df_pca_R = pd.DataFrame(result).T
df_pca_R.columns = ['R_PCA0','R_PCA1']
df_pca_R.head()
# df_pca_R.join(df.iloc[:,3:10]).corr()[(df_pca_R>0.4) | (df_pca_R<-0.4)]


# ## How does Q2 correlate with Q1?
# * On average at 0.71
# * But that's only from PCA0, not PCA1, ie. the main principal component.

# In[68]:


# Calculate average
df['mean_q2'] = 0.00
df['mean_q2'] = df.apply(lambda row: row[3:10].sum(), axis=1) / 7
# correlation
df[['mean_q2', 'Overall']].dropna().corr().iloc[0,1]


# In[62]:


# Correlations between Overall and pca vectors
df_pca_R.join(df[['Overall']]).dropna().corr().iloc[2,:2]


# ## What new information do we get from Q3?
# * Bathroom access was not taken into account by respondents as important overall - it loaded into the negative factor.
# * So, what are bathrooms like? We need a detailed question.
# * There are three PCA vectors - only the first one correlates with the overall rating of Bathrooms.

# In[41]:


from sklearn.decomposition import PCA
x = df.iloc[:,10:22].dropna()
pca_model = PCA(n_components=3) #  75% of variance with just two components, 86% with three.
pca_model.fit(x)
print(pca_model.explained_variance_ratio_)
print("Total variance explained is " + str(pca_model.explained_variance_ratio_.sum()))
print(pca_model.explained_variance_ratio_)


# In[36]:


# Loadings
pd.DataFrame(pca_model.components_.transpose(), index = x.columns).sort_values(0)


# In[47]:


# Calculate PCA values per observations
df_pca = pd.DataFrame(pca_model.components_.transpose(), index = x.columns)
result = []
for i in range(3):
    result.append(x.dot(df_pca.iloc[:,i]))
df_pca_B = pd.DataFrame(result).T
df_pca_B.columns = ['B_PCA0','B_PCA1', 'B_PCA2']


# In[64]:


# Obtain correlations
df_pca_B.join(df[['R_Bathrooms']]).dropna().corr().iloc[3,:3]


# In[ ]:




