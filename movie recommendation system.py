import numpy as np
import pandas as pd
movies=pd.read_csv("C:\\Users\\DELL'\\Desktop\\data science projects\\data science project datasets\\tmdb_5000_movies.csv")
credit=pd.read_csv("C:\\Users\\DELL'\\Desktop\\data science projects\\data science project datasets\\tmdb_5000_credits.csv")
films = movies.merge(credit, on='title')
films=films = films[['movie_id','title','overview','genres','keywords','cast','crew']]
films.dropna(inplace=True)
import ast
def convert(obj):
  l=[]
  for i in ast.literal_eval(obj):
    l.append(i['name'])
  return l
films['genres']=films['genres'].apply(convert)
films['keywords']=films['keywords'].apply(convert)
def convert3(obj):
  l=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter!=3:
      l.append(i['name'])
      counter+=1
    else:
      break
  return l
films['cast']=films['cast'].apply(convert3)
def fetch_director(obj):
  l=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      l.append(i['name'])
      break
  return l
films['crew']=films['crew'].apply(fetch_director)
films['overview'] = films['overview'].apply(lambda x: x.split())
films['genres']=films['genres'].apply(lambda x:[i.replace(" ","") for i in x])
films['keywords']=films['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
films['cast']=films['cast'].apply(lambda x:[i.replace(" ","") for i in x])
films['crew']=films['crew'].apply(lambda x:[i.replace(" ","") for i in x])
films['tags']=films['overview']+films['genres']+films['keywords']+films['cast']+films['crew']
films_df=films[['movie_id','title','tags']]
films_df['tags']=films_df['tags'].apply(lambda x:" ".join(x))
films_df['tags']=films_df['tags'].apply(lambda x:x.lower())
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)
films_df['tags']=films_df['tags'].apply(stem)
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(films_df['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
def recommend(movie):
  movie_index=films_df[films_df['title']==movie].index[0]
  distances=similarity[movie_index]
  movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movies_list:
    print(films_df.iloc[i[0]].title)
