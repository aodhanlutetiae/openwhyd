
# coding: utf-8

# note - there are 2 cells that need adjusting depending on which set (small sample, or real set) is being used:
# cell 73 & cell 376

# ISSUES
#  - if the video is unavailable, the scrape part of the algo returns a web address with just no track name. It needs to search again.

# ## import & wrangling

# In[1]:


import pandas as pd
import numpy as np
import random
import time
from bs4 import BeautifulSoup as bs
import requests


# In[65]:


path_to_file = "/Users/aidanairuser/Desktop/OPENWHYHD/openwhyhd_logs.csv"
sample = pd.read_csv(path_to_file)
print (sample.shape)
sample.sample(5)


# In[66]:


print ("nb of songs", sample.user.nunique())
print ("nb of users", sample.song.nunique())
print ("DATA TYPES")
print (sample.dtypes)

# there are 96 users in the sample set


# ## simplest rec: a top ten

# In[105]:


top_ten = sample.song.value_counts().head(11)
hits = pd.DataFrame(top_ten)
hits


# -- this is the top ten for the small sample set
# 
# scrape the details (see below):
# 
# 少年時代　井上揚水
# Green Velvet - I Want To Leave My Body
# 般若心経 托鉢 short ver  (heartsutra takuhatsu ver.) live act / 薬師寺寛邦(キッサコ)
# 般若心経 cho ver. (Heartsutra cho ver. ) live act / 薬師寺寛邦 キッサコ
# BEACH HOUSE -- "LEMON GLOW"
# Rejjie Snow - Egyptian Luvr (feat. Aminé & Dana Williams) (Official Audio)
# The Spinners - I'll Be Around
# ** the 8th most listened to is now 'video unavailable
# 6LACK - Never Know
# Best Music Mix 2017- Shuffle Music Video HD - Melbourne Bounce Music Mix 2017
# Underground Lovers - Your Eyes

# ## function to pull video title from site

# In[106]:



def find_toon(rem):
    
    top_ten_listens = []
    
    yt = "https://www.youtube.com/watch?v="
    url = yt.strip() + rem.strip()
    page = requests.get(url)
    soup = bs (page.text, 'html.parser')

    for item in soup.find_all('h1',{'class':'watch-title-container'}): 
        for post in item.find_all('span',{'class':'watch-title'}): 
            print (post.string.strip())
            print (url)
            
find_toon("PLACE_end_of_yt_address_here_please")


# ## pivot and flatten the log

# In[67]:


# flip sample to a multi-index / pivot table (aggfunc=len and aggfunc='count' both work as arguments)

freq = pd.pivot_table(sample,index=["song"],columns=["user"],values=["user"], aggfunc={"user":'count'})

freq.head(5)


# In[68]:


# turn selection into a df (multi-index)

pivoted = pd.DataFrame(freq)

print (pivoted.shape)

pivoted.head(5)


# In[73]:


# how sparse is the df? Print the total number of NaN and give as a %

count_nans = pivoted.isnull().sum().sum()
total_nb_cells = 80388 * 1275 
# total_nb_cells = 1688 * 96 # this line is for when we're using the sample set
perc = (count_nans/total_nb_cells)*100

print (count_nans, "cells are empty - soit", perc.round(decimals=2), "%", "of the df")


# In[74]:


####### ------------ RELOAD FROM HERE IF NECESSARY

# remove a dimension to get a df with regular columns

pivoted.columns = pivoted.columns.get_level_values(1)
pivoted_done = pd.DataFrame(pivoted)


# In[75]:


type(pivoted_done)


# In[76]:


pivoted_done.columns


# In[77]:


# replace the NaN with 0

pivoted_done.fillna(value = 0, inplace = True)


# In[101]:


# change column names from ints to "U6" etc.

string = "U"
pivoted_done.columns = [string+str(i) for i in range(1275)]


# In[102]:


# cast all entries to integers, not floats

pivoted_int = pivoted_done.astype(int, inplace = True)
pivoted_int.dtypes.head(5)

# add (int, errors = 'ignore') if that's what you need


# In[103]:


# THIS returns ONLY the integer - slice the row, then slice the column

pivoted_int.iloc[0]["U1"]


# In[104]:


pivoted_int.shape


# ## FINAL ALGORITHM (soir, mardi 19 mars)

# In[376]:



def filled_test(user_A, user_B):

    print ("The pair under consideration are", user_A, "and", user_B, "\n")

    # establish whether there are matches between the 2 users
    match_count_df = pivoted_int[[user_A, user_B]]
    a_b_match_count = len (match_count_df[(match_count_df[user_A] != 0) & (match_count_df[user_B] != 0)])

    if a_b_match_count == 0:
        print ("These two users have never listened to the same song, so they have nothing to suggest to the other", "\n")
    
    else:
    
        # cell is a variable so that we can increase the row by one each time
        row = 0
    
        # declare two empty lists, to store offers from one user to another
        b_TO_A_offer = []
        a_TO_B_offer = []

        # for x in range (1688): FOR USE WITH SMALL SAMPLE SET 
        for x in range (80388):
                  
            # find a song present in right, but not left, to be offered
            if pivoted_int.iloc[row][user_B] > 0 and pivoted_int.iloc[row][user_A] == 0:
                b_TO_A_offer += [row]
            
            # find a song present in left col, but not right, to be offered
            if pivoted_int.iloc[row][user_A] > 0 and pivoted_int.iloc[row][user_B] == 0:
                a_TO_B_offer += [row]
        
            row = row + 1
    
        print (a_b_match_count, "match(es) in total: Strength of match is TO BE DEFINED HERE PLEASE IN PART THROUGH % MATCH, IN PART THROUGH THE NB OF LISTENS TO EACH SONG \n")

        print (len (b_TO_A_offer), "offers for A col:", b_TO_A_offer, "\n")
        print (len (a_TO_B_offer), "offers for B col:", a_TO_B_offer, "\n")
 
        # pick from the list -- at random -- and make an offer for user A
        songA = random.choice(b_TO_A_offer)
        song_address = pivoted_int.index[songA]
        yt = "https://www.youtube.com/watch?v="
        completeA = yt.strip() + song_address.strip()
        print(user_A, "might like", completeA)
                      
        # pull video details from youtube for suggestion to user A - NOTE: 4 TIMES AS SLOW WITH THE TWO SCRAPES
        url = completeA
        page = requests.get(url)
        soup = bs (page.text, 'html.parser')
        for item in soup.find_all('h1',{'class':'watch-title-container'}): 
            for post in item.find_all('span',{'class':'watch-title'}): 
                details = post.string.strip()
                print (details, "\n")
    
        #  ... and for user B
        songB = random.choice(a_TO_B_offer)
        song_address = pivoted_int.index[songB]
        yt = "https://www.youtube.com/watch?v="
        completeB = yt.strip() + song_address.strip()
        print(user_B, "might like", completeB)
               
        # pull video details from youtube for suggestion to user B
        url = completeB
        page = requests.get(url)
        soup = bs (page.text, 'html.parser')
        for item in soup.find_all('h1',{'class':'watch-title-container'}): 
            for post in item.find_all('span',{'class':'watch-title'}): 
                details = post.string.strip()
                print (details, "\n")
    
    print ("HERE WE NEED TO FIND A WAY OF STORING UP THE DIFFERENT 'OFFERED TO U41' SERIES SO THAT WE CAN LOOK AT ALL THE SONGS BEING OFFERED TO B BY THE OTHER 95 USERS, AND CHOOSE THE MOST OFFERED", "\n")

    # this has been added to calculate the average nb of plays for each song a user listens to
    no_zeros_A = pivoted_int[(pivoted_int[user_A] != 0)] # all the full cells for the user column
    no_zeros_B = pivoted_int[(pivoted_int[user_B] != 0)]

    av_list_A = no_zeros_A[user_A].sum()/no_zeros_A[user_A].count() # a count (nb of songs) and a sum (nb of listens)
    av_list_B = no_zeros_B[user_B].sum()/no_zeros_B[user_B].count()

    print (user_A, "listened to each of his / her songs an average of", av_list_A.round(decimals=2), "times")
    print (user_B, "listened to each of his / her songs an average of", av_list_B.round(decimals=2), "times", "\n")


# ## run algo here:

# In[377]:


start_time = time.time()

# insert HERE the two users to be compared
filled_test("U2", "U3")

print("RAN IN --- %s seconds ---" % (time.time() - start_time))


# ## integrating the NUMBER of listens to a given song by a user to: 1 get a better picture of the strength of a match & 2 help get stronger recs

# In[374]:


# How do the users compare in the number of different songs they've listened to?

# no, I think this is total listens

pivoted_int.astype(bool).sum(axis=0).head(5)


# In[315]:


# find number of non-zero cells in a user's column, i.e. the number of songs they've listened to

print (pivoted_int.U26.astype(bool).sum(axis=0), "songs played by this user")


# In[309]:


# show all songs played, with number of listens for each

user_A = "U26"

non_zeros = (pivoted_int[(pivoted_int[user_A] != 0) ])

non_zeros[user_A]

