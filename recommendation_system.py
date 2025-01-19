import pandas as pd
from collections import defaultdict
import networkx as nx
from node2vec import Node2Vec

ratings = pd.read_csv(
    '/home/binit/Graph_neural_networks/ml-100k/u.data',
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'unix_timestamp']
)

movies = pd.read_csv(
    '/home/binit/Graph_neural_networks/ml-100k/u.item',
    sep='|',
    usecols=range(2),
    names=['movie_id', 'title'],
    encoding='latin-1'
)

ratings = ratings[ratings.rating >= 4]

pairs = defaultdict(int)

for _, group in ratings.groupby("user_id"):
    user_movies = list(group["movie_id"])
    for i in range(len(user_movies)):
        for j in range(i + 1, len(user_movies)):
            pairs[(user_movies[i], user_movies[j])] += 1

G = nx.Graph()

for pair in pairs:
    movie1, movie2 = pair
    score = pairs[pair]
    if score >= 20:
        G.add_edge(movie1, movie2, weight=score)

node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, p=2, q=1, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

def recommendation(movie_title):
    movie_id = movies[movies.title == movie_title].movie_id.values
    if len(movie_id) == 0:
        print("Movie not found.")
        return
    movie_id = str(movie_id[0])
    similar_movies = model.wv.most_similar(movie_id)[:5]
    for id, similarity in similar_movies:
        title = movies[movies.movie_id == int(id)].title.values[0]
        print(f'{title}: {similarity:.2f}')

recommendation('Star Wars (1977)')
