### movie.metadata.tsv Freebase'den alınmış 81.74 filmin metadata'sı. tür etiketleri burada.
### plot_summaries.txt ingilizce Wikipedia'dan çekilmiş 42.306 filmin özeti. Wikipedia ID'leri metadata'ya indeksli.

import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

pd.set_option('display.max_colwidth', 300) #max width to display

meta = pd.read_csv("movie.metadata.tsv", sep = '\t', header = None) #tsv as tap-separated-file, so sep=\t
#print(meta.head())

meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"] #renaming the columns that we will use later. the columns were shown with numbers
                                                             #but we are changing them into names

plots = [] #reading the plot summaries line by line and add them to the list

with open("plot_summaries.txt", 'r', encoding="utf8") as f: #for charmap error we specified encoding as utf8
       reader = csv.reader(f, dialect='excel-tab')
       for row in tqdm(reader):
            plots.append(row)

movie_id = [] #splitting movie id's and plot summaries into two separate lists
plot = []

for i in tqdm(plots):
  movie_id.append(i[0])
  plot.append(i[1])

movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot}) #creating a dataframe so we can handle the datas easily

#print(movies.head())

meta['movie_id'] = meta['movie_id'].astype(str) #changing movie id data type from integer to string

movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id') #merging specified columns of metadata with movies list
                                                                                      #which contains movie id and plot. merging done based on id's
#print(movies.head())

#print((movies['genre'][2])) #genres are in a dictionary structure but not a dictionary!!!!
                                    #we have to convert them into a list so we will work with them easily

genres = [] #we will extract all the genres from their id's and only show the genre names

for i in movies['genre']:
  genres.append(list(json.loads(i).values())) #for this we will use json loads. this parses string into dictionary so that we can use values

movies['genre_new'] = genres #we are adding the new parsed genre data to our main dataframe. after that this only shows genre names, not the id's

#print(movies.head())

#movies_new = movies[~(movies['genre_new'].str.len() == 0)] #this will remove all the movies that don't have any genre tag. this isn't used in our case

#print(movies_new.shape, movies.shape)

all_genres = sum(genres,[]) #a new separated list for genres. this will show the total number of all genres
#print(len(set(all_genres)))

all_genres = nltk.FreqDist(all_genres) #we will use this to have dictionary of genres and their frequence in the dataset

all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),
                              'Count': list(all_genres.values())})

#print(all_genres_df)

def clean_text(text): #we will clean our dataset with this function
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text) #remove everything except letters
    text = ' '.join(text.split())
    text = text.lower()

    return text

movies['clean_plot'] = movies['plot'].apply(lambda x: clean_text(x)) #we will use lambda function to clean plots

#print(movies.head())

def freq_words(x, terms): #this function will find the most used words in plot summaries
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    d = words_df.nlargest(columns="count", n=terms) #this will order the most frequent words by count

    #print(d)

#freq_words(movies['clean_plot'], 100) #we will find the most frequent 100 word

nltk.download('stopwords') #we will remove stop words from our data

stop_words = set(stopwords.words('english')) #set the language

def remove_stopwords(text): #this function will remove the stop words from our movies dataframe
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

movies['clean_plot'] = movies['clean_plot'].apply(lambda x: remove_stopwords(x))

freq_words(movies['clean_plot'], 100) #we will check the frequent words without stop words now

multilabel_binarizer = MultiLabelBinarizer() #we will convert genres into 0 and 1 binaries
multilabel_binarizer.fit(movies['genre_new'])

y = multilabel_binarizer.transform(movies['genre_new'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=40000) #tf-idf for feature extraction. this can be changed

xtrain, xval, ytrain, yval = train_test_split(movies['clean_plot'], y, test_size=0.2, random_state=9) #splitting data into train and validation

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain) #creating the tf-idf features
xval_tfidf = tfidf_vectorizer.transform(xval)

lr = LogisticRegression() #training
clf = OneVsRestClassifier(lr)

clf.fit(xtrain_tfidf, ytrain) #fitting the model on train data

y_pred = clf.predict(xval_tfidf) #prediction for validation set

#print(y_pred[3])

#print(multilabel_binarizer.inverse_transform(y_pred)[3])

# evaluate performance
f1_score(yval, y_pred, average="micro")

# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)

t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

# evaluate performance
f1_score(yval, y_pred_new, average="micro")

def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

for i in range(5):
  k = xval.sample(1).index[0]
  print("Movie: ", movies['movie_name'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",movies['genre_new'][k], "\n")

