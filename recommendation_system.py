import pandas as pd 
from collections import defaultdict

ratings = pd.read_csv('/home/binit/Graph_neural_networks/ml-100k/u.data', sep = '\t',
                      names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])
print(ratings.head(10))

movies = pd.read_csv('/home/binit/Graph_neural_networks/ml-100k/u.item', sep = '|',
                     usecols=range(2), names=['movie_id', 'title'],encoding='latin-1')

print(movies.head(10))

ratings = ratings[ratings.rating >= 4]
print(ratings)

pairs = defaultdict(int)
for group in ratings.groupby("userID"):
    user_movies = list(group[1]["movieId"])
    for i in range(len(user_movies)):
        for j in range(i+1, len(user_movies)):
            pairs[(user_movies[i], user_movies[j])]+=1
            
