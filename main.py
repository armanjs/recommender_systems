#import pandas and numpy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CounterVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load movies metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

#Credits, Genres, and Keywords Based Recommender

#load keywords and credits
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

#convert ids to int for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

#merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

#parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #check if more than 3 elements exist. If yes, return only first three, otherwise return entire list
        if len(names) > 3:
            names = names[:3]
        return names

        #return empty list in case of missing/malformed data
        return []

#define new director, cast, genres, and keywords features that are in a suitable form
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[features].apply(get_list)

#print the new features of the first 3 films
metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3)

#function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #check if director exists. if not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

#apply clean_data function to features
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

#joins all required columns by a space
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' '+ x['director'] + ' '+ ' '.join(x['genres'])

#create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)
metadata[['soup']].head(2)

#create count matrix using countervectorizer
count = CounterVectorizer(stop_words = 'english')
count_matrix = count.fit_transform(metadata['soup'])

count_matrix.shape

#compute cosine simliarity matrix based on count matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#reset index of your main dataframe and construct reverse mapping
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

def get_recommendations(title, cosine_sim=cosine_sim):
    #get index of movie corresponding to title
    idx = indices[title]

    #get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    #sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #get scores of 10 most similar movies
    sim_scores = sim_scores[1:11]

    #get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    #return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]
    
get_recommendations('Seventh Seal', cosine_sim2)
