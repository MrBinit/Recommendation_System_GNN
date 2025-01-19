import pandas as pd 

ratings = pd.read_csv('/home/binit/Graph_neural_networks/ml-100k/u.data', sep = '\t',
                      names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
print(ratings.head(10))

movies = pd.read_csv('/home/binit/Graph_neural_networks/ml-100k/u.item', sep = '|',
                     usecols=range(2), names=['movie_id', 'title'],encoding='latin-1')

print(movies.head(10))

ratings = ratings[ratings.rating >= 4]
print(ratings)