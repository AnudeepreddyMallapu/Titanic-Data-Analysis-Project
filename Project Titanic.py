#!/usr/bin/env python
# coding: utf-8

# # Project: Titanic Data Science Solution
# 
# ### Introduction
# 
# This dataset contains demographics and passenger information from 891 of the 2224 passengers and crew on board the Titanic. You can view a description of this dataset on the Kaggle website (https://www.kaggle.com/c/titanic), where the data was obtained.
# 
# In this analysis, I would like to explore the following questions.
# 
# 1.Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
# 
# 2.What deck were the passengers on and how does that relate to their class?
# 
# 3.Where did the passengers come from?
# 
# 4.Who was alone and who was with family?
# 
# 5.What factors helped someone survive the sinking?
# 
# 6.Was age a factor in determining the chances of survival?
# 
# 7.Did women had a better survival rate than men?
# 
# 8.How was children's survival rate as compared to men or women?
# 

# In[1]:


#importing Libraries
import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#import data from csv file
titanic_df= pd.read_csv(r"C:\Users\andee\Desktop\Titanic\train.csv")


# In[4]:


# Checking the number of rows and colmumns
titanic_df.shape


# In[5]:


titanic_df.head()


# In[6]:


#Overall info of the dataset
titanic_df.info()


# In[7]:


#checking gender
sns.countplot('Sex', data=titanic_df)


# In[8]:


# seperating the genders by classes
sns.countplot('Sex', data=titanic_df, hue='Pclass')


# In[9]:


sns.countplot('Pclass', data=titanic_df, hue='Sex')


# In[10]:


# We'll treat anyone as under 16 as a child, and then use the apply technique with a function to create a new column

# First let's make a function to sort through the sex 
def male_female_child(Passenger):
    age,sex = Passenger
    
    if age < 16:
        return 'child'
    else:
        return sex


# In[12]:


# Define a new column called 'person', remember to specify axis=1 for columns and not index
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[13]:


#checking the first ten rows
titanic_df[0:10]


# Now we have seperated the passengers between female,male,and child.

# In[14]:


sns.countplot('Pclass',data=titanic_df,hue='person')


# Interesting to see a lot of children in 3rd class and not so many in 1st!

# In[15]:


#Now, creating a distribution of the ages to get a more precise picture of the who the passengers were using pandas
titanic_df['Age'].hist(bins=70)


# In[16]:


#Now, getting mean of age
titanic_df['Age'].mean()


# In[17]:


# We could also get a quick overall comparison of male,female,child
titanic_df['person'].value_counts()


# In[20]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

# Set the figure equal to a facetgrid with the pandas dataframe as its data source, set the hue, and change the aspect ratio.
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

fig.map(sns.kdeplot, 'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim= (0,oldest))
fig.add_legend()


# In[19]:


import warnings
warnings.simplefilter('ignore')


# In[21]:


# We could have done the same thing for the 'person' column to include children:
fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)

fig.map(sns.kdeplot, 'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim= (0,oldest))
fig.add_legend()


# In[22]:


# Let's do the same for class by changing the hue argument:
fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot, 'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim= (0,oldest))
fig.add_legend()


# In[23]:


# Let's get a quick look at our dataset again
titanic_df.head()


# In[24]:


# First we'll drop the NaN values and create a new object, deck
deck = titanic_df['Cabin'].dropna()


# In[25]:


# Quick preview of the decks
deck.head()


# In[26]:


# So let's grab that letter for the deck level with a simple for loop
# Set empty list
levels=[]

# Loop to grab first letter
for level in deck:
    levels.append(level[0])
    
cabin_df= DataFrame(levels)

cabin_df.columns = ['Cabin']

sns.countplot('Cabin',data=cabin_df,palette='spring_d')


# Here we have a 'T' deck value there which doesn't make sense, we can drop it out with the following code:
# 

# In[28]:


# Redefine cabin_df as everything but where the row was equal to 'T'
cabin_df = cabin_df[cabin_df.Cabin != 'T']


# In[29]:


#Replot
sns.countplot('Cabin',data=cabin_df,palette='winter_d',order=['A','B','C','D','E','F','G'])


#  we've analyzed the distribution by decks
# 
# Now we have to answer,Where did the passengers come from?

# In[30]:


titanic_df.head(10)


# Note here that the Embarked column has C,Q,and S values. Reading about the project on Kaggle you'll note that these stand for Cherbourg, Queenstown, Southhampton.

# In[33]:


# Now, make a quick factorplot to check out the results, note the x_order argument, used to deal with NaN values
sns.countplot('Embarked',data=titanic_df,hue='Pclass',order=['C','Q','S'])


# An interesting find here is that in Queenstown, almost all the passengers that boarded there were 3rd class. It would be intersting to look at the economics of that town in that time period for further investigation.
# 

# Now finding Who was alone and who was with family?

# In[34]:


# Let's start by adding a new column to define alone

# add the parent/child column with the sibsp column
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df[0:10]


# Now we know that if the Alone column is anything but 0, then the passenger had family aboard and wasn't alone. So let's change the column now so that if the value is greater than 0, we know the passenger was with his/her family, otherwise they were alone.

# In[35]:


# Look for >0 or ==0 to set alone status

titanic_df['Alone'].loc[titanic_df['Alone']> 0] = 'Have Family'

titanic_df['Alone'].loc[titanic_df['Alone']==0] = 'Alone'


# In[36]:


# Check to make sure it worked
titanic_df.head()


# In[37]:


titanic_df['Alone']


# In[38]:


# Now let's get a simple visualization!
sns.countplot('Alone',data= titanic_df, palette = 'summer_d')


# 
# Great! Now that we analyzed the data let's go ahead and take a look at question: What factors helped someone survive the sinking?

# In[39]:


# Let's start by creating a new column for legibility purposes through mapping
titanic_df['Survivor'] = titanic_df.Survived.map({0:'No',1:'Yes'})
# Let's just get a quick overall view of survied vs died. 
sns.countplot('Survivor', data=titanic_df, palette='winter')


# So quite a few more people died than those who survived. Let's see if the class of the passengers had an effect on their survival rate.
# 
# 

# In[41]:


# use a factor plot again, but now considering class
sns.factorplot('Pclass','Survived',data=titanic_df)


# Look like survival rates for the 3rd class are substantially lower! But maybe this effect is being caused by the large amount of men in the 3rd class in combination with the women and children first policy. Let's use 'hue' to get a clearer picture on this.
# 
# 

# In[42]:


# Let's use a factor plot again, but now considering class and gender
sns.factorplot('Pclass','Survived', data=titanic_df,hue='person')


# From this data it looks like being a male or being in 3rd class were both not favourable for survival. Even regardless of class the result of being a male in any class dramatically decreases your chances of survival.
# 
# But what about age? Did being younger or older have an effect on survival rate?
# 
# 

# In[45]:


# Use a linear plot on age versus survival
sns.lmplot('Age','Survived',data=titanic_df)


# Looks like there is a general trend that the older the passenger was, the less likely they survived.
# 
# 

# In[46]:


# Let's use a linear plot on age versus survival using hue for class seperation
sns.lmplot('Age','Survived',hue= 'Pclass',data=titanic_df)


# We can also use the x_bin argument to clean up this figure and grab the data and bin it by age with a std attached!
# 
# 

# In[47]:


# Let's use a linear plot on age versus survival using hue for class seperation
generations=[10,25,40,55,70,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='summer',x_bins=generations)


# Interesting find on the older 1st class passengers! What about if we relate gender and age with the survival set?
# 
# 

# In[48]:


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='summer',x_bins=generations)


# In[49]:


sns.lmplot('Age','Survived',hue='Person', data=titanic_df, palette='inferno',x_bins=generations)


# Awesome! got some really great insights on how gender,age, and class all related to a passengers chance of survival.
# 
# 

# In[50]:


titanic_df.head()


# In[51]:


sns.factorplot('Cabin', data=cabin_df, palette='winter_d',kind='count')


# In[52]:


cabin_df.head()


# In[54]:


cabin_df = pd.concat([cabin_df, titanic_df['Sex']], axis=1)


# In[56]:


cabin_df = pd.concat([cabin_df, titanic_df['Survived']], axis=1)


# In[58]:


cabin_df.head()


# In[66]:


sns.factorplot('Survived',data=titanic_df, hue='Alone',palette='plasma',kind='count')


# In[68]:


sns.lmplot('SibSp','Survived',data=titanic_df, hue='Alone',palette='plasma')


# In[69]:


sns.lmplot('Parch','Survived',data=titanic_df, hue='Person',palette='plasma')


# In[70]:


sns.factorplot('Parch','Survived',data=titanic_df, hue='Person',palette='plasma')


# Conclsion:
# Analysis of this project mainly covered three factors in this anlysis (Age, Sex, Pclass).
# 
# Age: Doesn't play much role in determining the survival chances, except for ages below 1 years.
# Sex: Women had better chances of survival than men.
# 
# In general, Women & children across all classes had higer survival rates than men.
# 
# Pclass: Pclass-1 had best while Pclass-3 has the worst survival rate.
# 
# So we can say, that being a women in Pclass-1 seems to have the best chances of survival. However being a child or woman could not be considered as 100% survival chance.

# # References
# 
# https://www.kaggle.com/c/titanic/data
# 
# http://seaborn.pydata.org/generated/seaborn.factorplot.html#seaborn.factorplot
# 
# http://www.titanicfacts.net/titanic-victims.html

# In[ ]:




