import pandas as pd
from collections import defaultdict
import networkx as nx

ratings = pd.read_csv(
    '/home/binit/Graph_neural_networks/ml-100k/u.data',
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'unix_timestamp']
)
print(ratings.head(10))

movies = pd.read_csv(
    '/home/binit/Graph_neural_networks/ml-100k/u.item',
    sep='|',
    usecols=range(2),
    names=['movie_id', 'title'],
    encoding='latin-1'
)
print(movies.head(10))

ratings = ratings[ratings.rating >= 4]
print(ratings)

pairs = defaultdict(int)

#Increment counter specific to a pair of movies every time they are seen together in the same list
for _, group in ratings.groupby("user_id"):
    user_movies = list(group["movie_id"])
    for i in range(len(user_movies)):
        for j in range(i + 1, len(user_movies)):
            pairs[(user_movies[i], user_movies[j])] += 1
#till now it would store number of times two movies has been liked by the same user 

G = nx.Graph()

for pair in pairs:
    

