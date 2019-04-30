#!/usr/bin/env python
# coding: utf-8

# ### Résumé
# 
# Le log a environ 25 million données sur cinq ans - feb 2014 à avril 2019. Il s'agit d'environ
# 45K utilisateurs et 630K chansons. La procédure détaillé ci-dessous peut se résumer à:
# 
# - virer la colonne timestamp
# - faire un 'groupby' pour générer un tableau qui donne des lignes de chanson-utilisateur-nombre_d'écoutes
# - identifier et virer les liens morts
# - remanier le tableau en deux matrices utilisateur-chanson et chanson-utilisateur
# - paramètrer un modèle ALS en utilisant 'Implicit'
# - demander des chansons 'semblables' 
# - demander ensuite des 'chansons qui plairaient' à un utilisateur précis

# ## Résultat: on fournit un numéro d'utilisateur, et un nombre n, on reçoit n nombre de recommendations
# 
# Exemple: utilisateur 40, 5 chansons demandées
# 
# Exotica - Control Freak (Clip officiel)
# https://www.youtube.com/watch?v=98IslatTiws
# 
# Major Lazer - Aerosol Can ft. Pharrell Williams
# https://www.youtube.com/watch?v=pzZK4al4dvA
# 
# Ron Trent - You'll Never Find [preview]
# https://www.youtube.com/watch?v=fD_Wl-_Dvts
# 
# Parra for Cuva - Swept Away (feat. Anna Naklab & Mr. Gramo)
# https://www.youtube.com/watch?v=HpSCv8BjYzM
# 
# Phantogram "When I'm Small"
# https://www.youtube.com/watch?v=28tZ-S1LFok
# 

# ### imports et remanier le log pour qu'il soit au bon format

# In[1]:


# wrangling
import pandas as pd

# creating the csr matrices
import scipy.sparse as sparse 
from scipy.sparse.linalg import spsolve

# creating the ALS model
import implicit 
import numpy as np

# running the final 'find user' stage
from sklearn.preprocessing import MinMaxScaler

# scraping full song information from youtube
import requests # for scraping title from yt site
from bs4 import BeautifulSoup as bs # for retrieving info on the tracks from the youtube website.

# time the dead link checking function
import time

# play a sound when the function finishes
import os


# In[2]:


# importer le log (725 MB): 

open_sample = pd.read_csv('.CSV FILE LOCATION')


# In[3]:


# Regarder la forme, et la taille - les lignes sont par ordre chronologique (timestamp), non pas par utilisateur

print (open_sample.shape)
print ("nb of users:", open_sample.user.nunique())
print ("nb of songs:", open_sample.song.nunique())

open_sample.head(5)


# In[4]:


# On supprime la colonne 'timestamp' pour l'instant

open_sample.drop('timestamp', axis=1, inplace=True)


# In[5]:


# avec un groupby on remanie le df en 'user - song - number of plays'

flipped = pd.DataFrame(open_sample.groupby ( ['user', 'song']).song.count ( )).add_suffix('_count').reset_index()

flipped.rename(columns = {'user':'user_id'}, inplace = True)

print (flipped.shape)
print ("nb of users:", flipped.user_id.nunique())
print ("nb of diff songs:", flipped.song.nunique())

flipped.head(10)


# ## function to clean out dead links

# In[6]:


# Il n'y a que 631_348 chansons à vérifier -- non pas 8 millions -- tant qu'on vérifie pour les chansons 'uniques'

flipped.song.nunique()


# In[7]:


# on crée une liste des chansons uniques
song_list = flipped.song.unique()

# on crée un df depuis cette liste
checking = pd.DataFrame({'song':song_list})

# on rajoute une colonne vide pour noter 'ok' ou 'manque'
checking["link"] = ""

# on rajoute 'No' par défaut
checking['link'] = 'No'

len(checking)

# On a maintenant un df ou chaque chanson paraît une fois, avec 'No' indiqué dans la case 'lien fiable'


# ## reprendre avec une version antérieure qui est 'mi-vérifié'

# In[8]:


# Comme on nettoie le df en parties, parfois on reprend là où on s'était arrêté en important un df qu'on a déjà
# nettoyé en partie.

checking = pd.read_csv('.CSV FILE LOCATION')
checking.drop('Unnamed: 0', axis=1, inplace=True)
checking.head(5)


# In[ ]:


# Ici, on prend chaque ligne, on vérifie si le lien youtube est bon, on note, on continue. Dans son état ci-dessous
# on vérifie pour les lignes 210K à 220K.

start_time = time.time()

nb_ok = 0
row_start = 210_000
row_finish = 220_000
rows_total = row_finish - row_start

for row in range(row_start,row_finish):   
    toon = checking.iat[row, 0]
    yt = "https://www.youtube.com/watch?v="
    url = yt.strip() + toon.strip()
    page = requests.get(url)
    soup = bs (page.text, 'html.parser')

    for item in soup.find_all('h1',{'class':'watch-title-container'}): 
        for post in item.find_all('span',{'class':'watch-title'}): 
            checking.iat[row, 1] = 'yes'     
            nb_ok = nb_ok + 1
            row = row + 1   
            
print (nb_ok, 'are ok of', rows_total)

elapsed = ((time.time() - start_time)/60)
print ("ran in", elapsed, "minutes")

os.system('afplay /System/Library/Sounds/purr.aiff')
os.system('afplay /System/Library/Sounds/submarine.aiff')

# LOG OF TIME SPENT (usually a rate of 10 hours for 100K) and nb lost (about a sixth)

# 0 - 30_000: 25750 are ok of 30000 / ran in 305.8400292476018 minutes - 100 rows a minute
# 30_000 - 45_000: 12426 are ok of 15000 / ran in 153.5392414490382 minutes - 100 rows a minute
# 45_000 - 60_000: 12815 are ok of 15000 / ran in 150.01641809940338 minutes
# 60_000 - 70_000: 8564 are ok of 10000 / ran in 97.89230124553045 minutes
# 70_000 - 80_000: 8176 are ok of 10000 / ran in 93.95360571543375 minutes
# 80_000 - 90_000: 8281 are ok of 10000 / ran in 119.0353542526563 minutes
# 90_000 - 110_000: 16502 are ok of 20000 / ran in 210.82278453509014 minutes
# 110_00- 130_000: 16515 are ok of 20000 / ran in 211.21969944636027 minutes
# 130 to 170 : 32245 are ok of 40000 / ran in 395.40184058348336 minutes
# 170 to 200 : 24438 are ok of 30000 / ran in 298.89787494738897 minutes
# 200 to 210 : 8213 are ok of 10000 / ran in 107.48892344633738 minutes
# started 210K at 8.51


# In[10]:


# vérifier une trache de la df (là où ca passe de 'vérifié' à 'non vérifié', une fois les lignes / links vérifiés

checking.iloc[209_995:210_005,] 


# In[12]:


# téléchanger un fichier avec ce qu'on a vérifié jusqu'à là (histoire de ne pas recommencer du début la prochaine fois)

checking.to_csv("yt_log_checked_0_to_210K_0404.csv")


# In[ ]:





# ## reprendre avec fichier propre (ici c'est toujours avec liens morts pour l'instant)

# In[11]:


# changer le type de donnée pour 'song' d' "object" (string) à CATEGORIE
# créer une nouvelle colonne SONG_ID où chaque chanson à un ID de type (int 32)

flipped['song'] = flipped['song'].astype("category")

flipped['song_id'] = flipped['song'].cat.codes

flipped.head(5)


# In[12]:


# La Library IMPLICIT s'attend à des données dans deux matrices: ITEM-USER et USER-ITEM pour applique l'ALS.
# On utilise scipy pour construire ces deux matrices

sparse_song_user = sparse.csr_matrix((flipped['song_count'].astype(float), (flipped['song_id'], flipped['user_id'])))

sparse_user_song = sparse.csr_matrix((flipped['song_count'].astype(float), (flipped['user_id'], flipped['song_id'])))


# In[13]:


sparse_song_user


# In[14]:


sparse_user_song


# In[16]:


# Calculer le niveau de 'sparseness' de la matrice -- elle est très élevée

sparse_song_user_filled_cells = 8741735
sparse_song_user_total_cells = 45904*631348
sparse_song_user_empty_cells = sparse_song_user_total_cells - sparse_song_user_filled_cells

level_of_sparseness_SU = (sparse_song_user_empty_cells/sparse_song_user_total_cells) * 100
print ("level of sparseness for song_user:", level_of_sparseness_SU)


# ## Maintenant qu'on a les deux matrices, on passe à l'algorithm ALS depuis Implicit

# In[18]:


# Définir notre version du modèle ALS

model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)


# In[19]:


# La mesure Confiance dans l'aglo a besoin d'un valeur ALPHA pour marcher. Ici, on donne 15.

alpha_val = 15

# La mesure de confiance est calculé: matrice SONG_ITEM x confiance

data_conf = (sparse_song_user * alpha_val).astype('double')

# La modèle est 'fitté' avec cette mesure de confiance

model.fit(data_conf)


# In[20]:


# now that 'model' is running, we can call methods on it

user_vecs = model.user_factors

song_vecs = model.item_factors


# In[21]:


# Using the SONG vector (= a kind of profile) as an argument "Calculate the vector norms"

song_norms = np.sqrt((song_vecs * song_vecs).sum(axis=1))


# ### Maintenant qu'on a les 2 matrices et l'algo ALS - on précise une chanson

# In[26]:


# Prenons la chanson X...
# trouvons N nombre de chansons 'semblables' (chanson d'origine comprise, donc n_similar = 10 nous rend neuf propositions)

find_song_id = 1800

n_similar = 10


# In[27]:


# Calculate the similarity score, grab the top N items and create list of song-score tuples of most similar songs

# (this is where the mysterious 'song_id' intervenes - also where the 'dot products' are measured compared)

scores = song_vecs.dot(song_vecs[find_song_id]) / song_norms

top_idx = np.argpartition(scores, -n_similar)[-n_similar:]

similar = sorted(zip(top_idx, scores[top_idx] / song_norms[find_song_id]), key=lambda x: -x[1])


# In[33]:


# Imprimer les chansons: celle qu'on a fourni, et les propositions qui y répondent. Les mettre dans une liste

songs_propd = []

for item in similar:
    idx, score = item
    found = flipped.song.loc[flipped.song_id == idx].iloc[0]
    songs_propd.append(found)
    print (found)

print (songs_propd)


# In[29]:


# on refait tourner mais avec un YT scrape / lookup pour pouvoir écouter 

print ("LA CHANSON QU'ON A FOURNIE C'EST LA PREMIERE. CELLES QUI SUIVENT ONT ÉTÉ RETROUVÉES")

for item in similar:    
    idx, score = item
    sugg = flipped.song.loc[flipped.song_id == idx].iloc[0]
        
    yt = "https://www.youtube.com/watch?v="
    url = yt.strip() + sugg.strip()
    page = requests.get(url)
    soup = bs (page.text, 'html.parser')

    for item in soup.find_all('h1',{'class':'watch-title-container'}): 
        for post in item.find_all('span',{'class':'watch-title'}): 
            print (post.string.strip())
            print (url)


# ## Create USER RECOMMENDATIONS

# In[34]:


# cette fonction prend: utilisateur_ID, matrice USER_SONG, user_vecs & song_vecs (qu'on vient de créer), 
# il rend un modèle 'recommendations'

# NOTONS - c'est ici qu'on précise qu'on cherche CINQ (ou deux, ou vingt) recommendations

def recommend(user_id, sparse_user_song, user_vecs, song_vecs, num_items=5):

    user_interactions = sparse_user_song[user_id,:].toarray()

    user_interactions = user_interactions.reshape(-1) + 1
    
    user_interactions[user_interactions > 1] = 0

    rec_vector = user_vecs[user_id,:].dot(song_vecs.T).toarray()

    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions * rec_vector_scaled

    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    songs = []
    scores = []

    for idx in item_idx:
        
        songs.append(flipped.song.loc[flipped.song_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'song': songs, 'score': scores})

    return recommendations


# In[35]:


# Get the trained user and item vectors. We convert them to csr matrices to work with our previous recommend function.
# we ALREADY had user and song_vecs - but here they're defined in terms of the matrices.

# On précise un utilisateur - on demande des recommendations

user_vecs = sparse.csr_matrix(model.user_factors)
song_vecs = sparse.csr_matrix(model.item_factors)

#item_vecs = model.item_factors

# Create recommendations for user with id 40
user_id = 40

recommendations = recommend(user_id, sparse_user_song, user_vecs, song_vecs)

print (recommendations)


# In[37]:


# on refait tourner mais avec un YT scrape / lookup pour pouvoir écouter

rec_songs_only = recommendations.song
happy_listener = user_id

print ("LA CHANSON QU'ON A FOURNIE C'EST LA PREMIERE. CELLES QUI SUIVENT ONT ÉTÉ RETROUVÉES")
print ("l'auditeur content c'est le", happy_listener)
print ("S'il n'y a pas", len (recommendations), "chansons ci-dessous, c'est qu'il y en a qui n'existent plus sur YT")

for item in rec_songs_only:    
        
    yt = "https://www.youtube.com/watch?v="
    url = yt.strip() + item.strip()
    page = requests.get(url)
    soup = bs (page.text, 'html.parser')

    for item in soup.find_all('h1',{'class':'watch-title-container'}): 
        for post in item.find_all('span',{'class':'watch-title'}): 
            print (post.string.strip())
            print (url)



# In[ ]:





# In[ ]:





# ### Annexe

# In[27]:


# la fonction de base pour retrouver titre et lien d'une chanson

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
            
find_toon("-CNrGqHKoa8")

