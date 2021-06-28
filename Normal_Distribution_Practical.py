#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[2]:


inf_df = pd.read_csv("Downloads\\INFY.NS.csv")


# In[3]:


tat_df = pd.read_csv("Downloads\\TATAMOTORS.NS.csv")


# In[8]:


inf_df.head()


# In[7]:


tat_df.head()


# Since our analysis involve only daily close prices, We will select date and close columns

# In[10]:


inf_df = inf_df[['Date','Close']]


# In[11]:


tat_df = tat_df[['Date','Close']]


# In[12]:


inf_df.head()


# In[13]:


tat_df.head()


# Visualizing the daily close prices will show how stock prices have moved over time. So, I will change the date column into DateTimeIndex

# In[14]:


inf_df = inf_df.set_index(pd.DatetimeIndex(inf_df['Date']))
tat_df = tat_df.set_index(pd.DatetimeIndex(tat_df['Date']))


# In[15]:


inf_df


# In[16]:


tat_df


# Plot the trend of close prices

# In[19]:


plt.plot(inf_df.Close);
plt.xlabel('Time');
plt.ylabel('Close Price-INFOSYS');


# In[20]:


plt.plot(tat_df.Close);
plt.xlabel('Time');
plt.ylabel('Close Price-TATA Motors');


# # Observations:-
# There is an upward trend in Infosys share price but in March-2021 during the covid-19 outbreak, it went down and recovered exponentially.
# But TATA Motor's share price is very volatile and there is a downward trend since 2017. This is because their profit and market share went down.

# # But as an intraday trader, I would like to know the following questions:-
# 
# 1.What is the expected daily rate of return?
# 
# 2.Which stocks have higher risk or volatility as far as the daily return is concerned?
# 
# 3.What is the expected range of return for 95% confidence interval?
# 
# 4.Which stock has higher profitability of making a daily return of 2% or more?
# 
# 5.which stock has higher profitability of making a loss(risk) of 2% or more?
# 

# In[23]:


#What is the expected daily rate of return?

inf_df['gain'] = inf_df.Close.pct_change(periods = 1)
tat_df['gain'] = tat_df.Close.pct_change(periods = 1)


# In[24]:


inf_df.head()


# In[25]:


tat_df.head()


# In[26]:


#droping NAN Values
inf_df = inf_df.dropna()
tat_df = tat_df.dropna()


# In[27]:


inf_df.head()


# In[28]:


tat_df.head()


# Now, plot gain against time

# In[29]:


plt.figure(figsize = (8, 6))
plt.plot(inf_df.index, inf_df.gain);
plt.xlabel('Time');
plt.ylabel('gain');


# In[31]:


plt.figure(figsize = (8, 6))
plt.plot(tat_df.index, tat_df.gain);
plt.xlabel('Time');
plt.ylabel('gain');


# The daily gain is highly random and fluctuates around 0.The gain remains mostly between 0.05 and -0.05. However, very high gain in in infosys is close to 17 and loss is 22.

# In[32]:


sns.distplot(inf_df.gain, label = 'Infosys');
sns.distplot(tat_df.gain, label = 'Tata Motors');
plt.xlabel('Gain');
plt.ylabel('Density');
plt.legend();


# From the above plot gain seems to be normally distributed for both the stocks with mean around 0.
# Tata motors seems have a higher variance than Infosys.

# # Mean and Variance

# In[37]:


print("Daily gain of Infosys")
print("---------------------")
print("Mean: ",round(inf_df.gain.mean(),4))
print("Standard Deviation: ",round(inf_df.gain.std(),4))


# In[39]:


print("Daily gain of Tata Motors")
print("-------------------------")
print("Mean: ",round(tat_df.gain.mean(),4))
print("Standard Deviation: ",round(tat_df.gain.std(),4))


# The describe() method of DataFrame returns the detailed statistical summary of variable

# In[40]:


inf_df.gain.describe()


# In[41]:


tat_df.gain.describe()


# The standard deviation of infosys is 0.018292 and for Tata motors 0.027387, thus tata motors stock is more risky than Infosys. The mean of both the stocks are close to 0.

# # Confidence Interval

# alpha: 0.05
# 
# loc: Mean for normal distribution
# 
# scal: Standard Deviation for normal distribution

# In[45]:


from scipy import stats
import numpy as np


# In[46]:


inf_df_ci = stats.norm.interval(0.95, loc = inf_df.gain.mean(),  scale = inf_df.gain.std())

print("Gain at 95% confidence interval is: ",np.round(inf_df_ci,4))


# In[47]:


tat_df_ci = stats.norm.interval(0.95, loc = tat_df.gain.mean(),  scale = tat_df.gain.std())

print("Gain at 95% confidence interval is: ",np.round(tat_df_ci,4))


# # Question 4 & 5 can be answered using cumulative distribution function

# In[49]:


print("Probability of making 2% loss or higher in Infosys: ")
np.round(stats.norm.cdf(-0.02, loc = inf_df.gain.mean(),
               scale = inf_df.gain.std()),4)


# In[50]:


print("Probability of making 2% loss or higher in Tata Motors: ")
np.round(stats.norm.cdf(-0.02, loc = tat_df.gain.mean(),
               scale = tat_df.gain.std()),4)


# In[52]:


print("Probability of making 2% gain or higher in Infosys: ")
np.round(1-stats.norm.cdf(0.02, loc = inf_df.gain.mean(),
               scale = inf_df.gain.std()),4)


# In[53]:


print("Probability of making 2% gain or higher in Tata Motors: ")
np.round(1-stats.norm.cdf(0.02, loc = tat_df.gain.mean(),
               scale = tat_df.gain.std()),4)

