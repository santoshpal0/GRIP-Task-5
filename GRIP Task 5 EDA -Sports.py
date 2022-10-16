#!/usr/bin/env python
# coding: utf-8

# ## Task 5: Exploratory Data Analysis on dataset 'Indian Premier League'
# ## Objective
# ### Performing "Exploratory Data Analysis" on the dataset "Indian Premier League".
# ### As a sports analysts, find the most successful teams, players and factors contributing win or loss of a team.
# ### Suggest teams or players a company should endorse for its products. 

# ### Import Necessary Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt              
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
import warnings                              
warnings.filterwarnings("ignore")


# ### Load the Data

# In[56]:


# Loading the "matches" dataset
matches=pd.read_csv("C:\\Users\\user\\Downloads\\matches.csv")
scores=pd.read_csv("C:\\Users\\user\\Downloads\\deliveries.csv")
scores


# In[5]:


# import first 5 rows
matches.head()


# In[7]:


deliveries.head()


# # checking dimension (num of rows and columns) of dataset
# matches.shape

# In[9]:


# check dataframe structure like columns and its counts, datatypes & Null Values
matches.info()


# In[10]:


matches.columns


# In[11]:


matches.dtypes.value_counts()


# In[12]:


# Gives number of data points in each variable
matches.count()


# In[13]:


# descriptive statistics
matches.describe()


# In[14]:


matches.isnull().sum()


# ### Team won by Maximum runs

# In[8]:


matches.iloc[matches['win_by_runs'].idxmax()]


# In[9]:


matches.iloc[matches['win_by_runs'].idxmax()]['winner']


# ### Team Won by Maximum Wickets

# In[11]:


matches.iloc[matches['win_by_wickets'].idxmax()]['winner']


# ### Team Won by Minimum Runs

# In[15]:


matches.iloc[matches[matches['win_by_runs'].ge(1)].win_by_runs.idxmin()]['winner']


# ### Team Won by Minimum Wickets 

# In[16]:


matches.iloc[matches[matches['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]


# In[18]:


matches.iloc[matches[matches['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]['winner']


# ## Observations 

# #### 1. Mumbai Indians is the team which won by maximum and minimum runs
# #### 2.Kolkata Knight Riders is the team which won by maximum and minimum wickets

# ###  Season Which had most number of matches

# In[20]:


plt.figure(figsize=(12,6))
sns.countplot(x='season', data=matches)
plt.show()


# #### In  2013 we have the most number of matches

# In[27]:


plt.figure(figsize=(12,6))
data = matches.winner.value_counts()
sns.barplot(y = data.index, x = data, orient='h')
plt.show()


# #### Mumbai Indians is the winners in most of the matches 

# ### Top Players of the match winners

# In[29]:


top_players = matches.player_of_match.value_counts()[:10]
#sns.barplot(x="day", y="total_bill", data=df)
fig, ax = plt.subplots(figsize=(15,8))
ax.set_ylim([0,20])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
top_players.plot.bar()
sns.barplot(x = top_players.index, y = top_players, orient='v', palette="Blues");
plt.show()


# #### CH Gayle is the most successful player in all match winners 

# #### Number of Matches in each venue: 

# In[31]:


plt.figure(figsize=(12,6))
sns.countplot(x='venue', data=matches)
plt.xticks(rotation='vertical')
plt.show()


# ### Eden Gardens is the venue where largest number of matches were played

# ###  Number of matches played by each team:

# In[32]:


temp_df = pd.melt(matches, id_vars=['id','season'], value_vars=['team1', 'team2'])

plt.figure(figsize=(12,6))
sns.countplot(x='value', data=temp_df)
plt.xticks(rotation='vertical')
plt.show()


# ### "Mumbai Indians" lead the pack with most number of matches played followed by "Royal Challengers Bangalore". 

# #### Number of wins per team:

# In[34]:


plt.figure(figsize=(12,6))
sns.countplot(x='winner', data=matches)
plt.xticks(rotation=90)
plt.show()


# #### MI again leads the pack followed by CSK. 

# ####  Champions each season:
# 
# Now let us see the champions in each season.

# In[35]:


temp_df = matches.drop_duplicates(subset=['season'], keep='last')[['season', 'winner']].reset_index(drop=True)
temp_df


# ### Toss Decision

# In[37]:


temp_series = matches.toss_decision.value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss decision percentage")
plt.show()


# ####  Almost 55% of the toss decisions are made to field first. Now let us see how this decision varied over time

# In[39]:


plt.figure(figsize=(12,6))
sns.countplot(x='season', hue='toss_decision', data=matches)
plt.xticks(rotation='vertical')
plt.show()


# #### Look at 2014 and after all teams want to field first 

# In[41]:


# Since there is a very strong trend towards batting second let us see the win percentage of teams batting second.
num_of_wins = (matches.win_by_wickets>0).sum()
num_of_loss = (matches.win_by_wickets==0).sum()
labels = ["Wins", "Loss"]
total = float(num_of_wins + num_of_loss)
sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win percentage batting second")
plt.show()


# ### So percentage of times teams batting second has won is 53.2. Now let us split this by year and see the distribution.

# ###  Top players of the match:

# In[42]:


# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%d' % int(height),
                ha='center', va='bottom')


# In[44]:


temp_series = matches.player_of_match.value_counts()[:10]
labels = np.array(temp_series.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_series), width=width)
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top player of the match awardees")
autolabel(rects)
plt.show()


# ### CH Gayle is the top player of the match awardee in all the seasons of IPL.

# ### Top Umpires: 

# In[46]:


temp_df = pd.melt(matches, id_vars=['id'], value_vars=['umpire1', 'umpire2'])

temp_series = temp_df.value.value_counts()[:10]
labels = np.array(temp_series.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_series), width=width,)
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top Umpires")
autolabel(rects)
plt.show()


# ### S Ravi has done umpiring in large number of matches

# ###  Score Data Set

# In[57]:


scores.head()


# ###  Batsman analysis:

# ####  Let us start our analysis with batsman. Let us first see the ones with most number of IPL runs

# In[60]:


temp_df = scores.groupby('batsman')['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='blue')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top run scorers in IPL")
ax.set_xlabel('Batsmane Name')
autolabel(rects)
plt.show()


# #### Virat Kohli is leading the chart followed closely by Raina

# In[62]:


# Now let us see the players with more number of boundaries in IPL.
temp_df = scores.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='lightskyblue')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Batsman with most number of boundaries.!",fontsize = 10)
autolabel(rects)
plt.show()


# #### S Dhawan hits the most boundaries followed by SK Raina

# In[64]:


# Now let us check the number of 6's
temp_df = scores.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='m')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation=90)
ax.set_ylabel("Count")
ax.set_title("Batsman with most number of sixes.!")
ax.set_xlabel('Batsmane Name')
autolabel(rects)
plt.show()


# #### CH Gayle hit the most number of sixes 

# In[65]:


# Now let us see the batsman who has played the most number of dot balls.
temp_df = scores.groupby('batsman')['batsman_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='c')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Batsman with most number of dot balls.!")
ax.set_xlabel('Batsmane Name')
autolabel(rects)
plt.show()


# #### V Kohli with most number of dot balls 

# In[66]:


# Let us check the percentage distribution now.
def balls_faced(x):
    return len(x)

def dot_balls(x):
    return (x==0).sum()

temp_df = scores.groupby('batsman')['batsman_runs'].agg([balls_faced, dot_balls]).reset_index()
temp_df = temp_df.loc[temp_df.balls_faced>200,:]
temp_df['percentage_of_dot_balls'] = (temp_df['dot_balls'] / temp_df['balls_faced'])*100.
temp_df = temp_df.sort_values(by='percentage_of_dot_balls', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

fig, ax1 = plt.subplots(figsize=(15,8))
ax2 = ax1.twinx()
labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
rects = ax1.bar(ind, np.array(temp_df['dot_balls']), width=width, color='brown')
ax1.set_xticks(ind+((width)/2.))
ax1.set_xticklabels(labels, rotation='vertical')
ax1.set_ylabel("Count of dot balls", color='brown')
ax1.set_title("Batsman with highest percentage of dot balls (balls faced > 200)")
ax2.plot(ind+0.45, np.array(temp_df['percentage_of_dot_balls']), color='b', marker='o')
ax2.set_ylabel("Percentage of dot balls", color='b')
ax2.set_ylim([0,100])
ax2.grid(b=False)
plt.show()


# #### SG Ganguly with highest percentage of dot balls

# ### Bowler Analysis 

# In[67]:


#Now let us see the bowlers who has bowled most number of balls in IPL.
temp_df = scores.groupby('bowler')['ball'].agg('count').reset_index().sort_values(by='ball', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['bowler'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['ball']), width=width, color='cyan')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top Bowlers - Number of balls bowled in IPL")
ax.set_xlabel('Bowler Names')
autolabel(rects)
plt.show()


# ####  Harbhajan Singh is the the bowler with most number of balls bowled in IPL matches.

# In[69]:


#Bowler with most number of dot balls
temp_df = scores.groupby('bowler')['total_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='total_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['bowler'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['total_runs']), width=width, color='lightskyblue')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top Bowlers - Number of dot balls bowled in IPL")
ax.set_xlabel('Bowler Names')
autolabel(rects)
plt.show()


# #### Harbhajan Singh bowled most number of dot balls 

# In[71]:


# Now let us see the bowlers who has bowled more number of extras in IPL.
temp_df = scores.groupby('bowler')['extra_runs'].agg(lambda x: (x>0).sum()).reset_index().sort_values(by='extra_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['bowler'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_df['extra_runs']), width=width, color='magenta')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Bowlers with more extras in IPL")
ax.set_xlabel('Bowler Names')
autolabel(rects)
plt.show()


# #### SL Malinga gives more number of extras 

# In[72]:


# Now let us see most common dismissal types in IPL.
plt.figure(figsize=(12,6))
sns.countplot(x='dismissal_kind', data=scores)
plt.xticks(rotation='vertical')
plt.show()


# ####  Caught is the most common dismissal type in IPL followed by Bowled

# In[ ]:




