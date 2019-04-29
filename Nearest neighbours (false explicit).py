#!/usr/bin/env python
# coding: utf-8

# ## Résumé - outils et résultats

# Ici on cherche des récommendations sur le log des chansons écoutées, mais suite à un remaniement qui attribue une 
# note (entre 0 et 5) à chaque chanson pour chaque utilisateur. (Remaniement effectué par https://github.com/Patouche). Ce remaniement semble avoir rassemblé les 'doublons' (doublon: un utilisateur écoute chanson x un jour, le réécoute le lendemain) pour que chaque paire utilisateur--chanson paraisse uniquement maintenant une fois, avec son score.
# 
# OUTILS: library Surprise
# 
# RESULTATS
# 
# 1. KNNwithMeans - plus des utilisateurs se rassemblent dans leur consomation, plus ils devraient s'échanger de chansons.
# On arrive à prédire un score pour une chanson, dans le context d'un utilisateur, et sur (je crois) une échelle de 0 à 5, genre (pour une chanson très appréciée, et une autre peu appréciée):
# user 0 & chanson 55915: 4.52 est 
# user 0 & chanson 78257: 0.37 est
# - mais comme ces données viennent comme un élément à l'intérieur d'un object Surprise, je ne vois pas encore 
# comment exploiter cet outil. Il faudrait d'abord isoler le score 'est' d'une longue chaîne de résultats, et ensuite...mettre un boucle à tourner pour prédire chaque score pour un utilisateur? Très peu efficace.
# 
# 2. Une fois les ensembles 'train' et 'test' établis à partir de notre log, on peut comparer des différents algos
# pris de Surprise. On a simplement pris KNNwithMeans mais apparemment c'est BaselineOnly qui a le plus petit RMSE
# 

# ## importer et remanier

# In[36]:



import csv
import pandas as pd

from surprise import KNNBasic
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline

from surprise import Dataset
from surprise import Reader

from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split


# In[37]:


# import the log with its 'scores' derived by patouche
log = pd.read_csv('/Users/aidanairuser/Desktop/OPENWHYD/users_rating_pat.csv')

# rename the 'count' column because 'count' is a method in python libraries
log.rename(columns = {'song_id':'song','count':'count_score'}, inplace = True)

print(log.shape)
log.head(5)


# In[38]:


# changer le type de la colonne 'song' à 'catégorie'
log['song'] = log['song'].astype("category")

# créer une colonne 'song ID' qui sera plus facile à manipuler que les strings youtube qui font 'nom'
log['song_id'] = log['song'].cat.codes

# changer le type de song_id aussi pour que la colonne soit 'catégorie'
log.song_id = log.song_id.astype("category")


# In[39]:


log.info()


# In[40]:


# remanier l'ordre des colonnes pour mieux suivre la logique qui sera appliquée plus bas

log = log[['user_id', 'song_id', 'count_score']]


# In[41]:


log.head(3)


# ## 1. Kerpanic KNNwithMeans (from surprise) - explicit User User collaborative filtering
# 
# https://kerpanic.wordpress.com/2018/03/26/a-gentle-guide-to-recommender-systems-with-surprise/
# 
#     USER:USER
#     The output is the prediction of user u’s rating on item i:
#     si on donnait chanson X à utilisateur, que dirait elle?
#     We utilize the similarity measure between user u and user v in this case.
#     
#     ITEM:ITEM - DO NOT RUN, TAKES FOREVER / HANGS. I DONT KNOW WHY
#     Instead of using user similarity, we use item similarity measure to calculate the prediction.
#     Ssimilarity is now between item i and item j, instead of user u and v as before.
# 

# In[42]:



# on crée / définit un lecteur en précisant l'échelle de notes
reader = Reader(rating_scale = (0, 5))

# on définit 'data' qui prendra comme paramètres les colonnes utilisateur, chanson et score (et le lecteur)
data = Dataset.load_from_df(log[['user_id', 'song_id', 'count_score']], reader=reader)

# on divise le data pour garder 15% pour la partie test
trainset, testset = train_test_split(data, test_size=.15)

# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})

# on utilise la partie 'training' pour former l'algo sur notre log
algo.fit(trainset)


# In[43]:


# avec l'algo 'formé' sur notre log, on peut demander des prédictions en précisant des utilisateurs et 
# chansons et en utilisant algo.predict

uid = 1273  # utilisateur 102
iid = 73445  # chanson 2

# get a prediction for specific users and items.

pred = algo.predict(uid, iid, r_ui=4, verbose = True)
pred

# résultat: estimé 0.57


# notons les scores rendu par l'algo quand on demande des chansons aimé / mal aimé par les utilisateurs
# 
# user 0 / chanson 55915 (score 5) gives 4.52 est
# user 0 / chanson 78257 (score 1) gives 0.37 est
# user 1270 / chanson 18125 (score 5) gives 2 est
# user 1273 / chanson 73445 (score 5) gives 5 est
# user 1274 / chanson 61382 (score 0) gives 0.43 est
# user 1274 / chanson 60518 (score 5) gives 5 est
# 

# In[ ]:





# ## 2. choisir son algo - d'abord notre KNNwithMeans

# In[30]:



# on crée / définit un lecteur en précisant l'échelle de notes
reader = Reader(rating_scale = (0, 5))

# on définit 'data' qui prendra comme paramètres les colonnes utilisateur, chanson et score (et le lecteur)
data = Dataset.load_from_df(log[['user_id', 'song_id', 'count_score']], reader=reader)

# on effectue une 'cross validation' avec ce 'data' et l'algo KNN basic
cross_validate(KNNWithMeans(), data, verbose=False)


# ## ensuite les autre...

# In[31]:


# KNN BASIC

cross_validate(KNNBasic(), data, verbose=False)


# In[32]:


# KNN BASELINE

cross_validate(KNNBaseline(), data, verbose=False)


# In[35]:


# KNN WITH Z SCORE

cross_validate(KNNWithZScore(), data, verbose=False)


# In[33]:


# NORMAL PREDICTOR

cross_validate(NormalPredictor(), data, verbose=False)


# In[34]:


# BASELINE ONLY

cross_validate(BaselineOnly(), data, verbose=False)


# In[ ]:




