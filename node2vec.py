from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import networkx as nx
import numpy as np
import random

G = nx.karate_club_graph()
labels = []
for node in G.nodes():
    label = G.nodes[node]['club']
    labels.append(1 if label == 'Officer' else 0)

walks = []

def next_node(previous, current, p, q):
    neighbors = list(G.neighbors(current))
    alphas = []
    for neighbor in neighbors:
        if neighbor == previous:
            alpha = 1 / p
        elif G.has_edge(neighbor, previous):
            alpha = 1
        else:
            alpha = 1 / q
        alphas.append(alpha)
    probs = [alpha / sum(alphas) for alpha in alphas]
    next_node = np.random.choice(neighbors, size=1, p=probs)
    return next_node[0]

def random_walk(start, length, p=1, q=1):
    walk = [start]
    for _ in range(length - 1):
        current_node = walk[-1]
        previous = walk[-2] if len(walk) > 1 else None
        next_n = next_node(previous, current_node, p, q)
        walk.append(next_n)
    return [str(x) for x in walk]

for node in G.nodes():
    for _ in range(80):
        walks.append(random_walk(node, 10, 3, 2))

node2vec = Word2Vec(
    walks,
    hs=1,
    sg=1,
    vector_size=100,
    window=10,
    workers=2,
    min_count=1,
    seed=0
)

node2vec.train(walks, total_examples=node2vec.corpus_count, epochs=30, report_delay=1)

train_mask = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
train_mask_str = [str(x) for x in train_mask]
test_mask = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33]
test_mask_str = [str(x) for x in test_mask]
labels = np.array(labels)

clf = RandomForestClassifier(random_state=0)
clf.fit([node2vec.wv[x] for x in train_mask_str], labels[train_mask])

y_pred = clf.predict([node2vec.wv[x] for x in test_mask_str])
acc = accuracy_score(y_pred, labels[test_mask])
print(f'Node2Vec accuracy = {acc * 100:.2f}%')
