"""
Authors: Magdalena Asmus-Mrzygłów, Patryk Klimek
required libraries
"""
import json

import numpy as np


def euclidean_score(dataset, user1, user2):
    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users,
    # then the score is 0
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def movie_suggestor(data, user1):
    if user1 not in data:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    users_list = list(data.keys())
    users_list.remove(user1)

    score_list = {name: euclidean_score(data, user1, name) for name in users_list
                  if euclidean_score(data, user1, name) != 0}

    smallest_distance_person = sorted(score_list.items(), key=lambda x: x[1])[0][0]


    sorted_movies = sorted(data[smallest_distance_person].items(), key=lambda x: x[1], reverse=True)
    suggested_movies = [movie for movie, score in sorted_movies[0:5]]
    not_suggested_movies = [movie for movie, score in sorted_movies[-6:-1]]

    print('Suggested movies:')
    print(suggested_movies)
    print('Not suggested movies:')
    print(not_suggested_movies)



if __name__ == '__main__':
    # define name of file with data and person for whom recommendation will be made
    rating_file = 'movies_data.json'
    user = input("Podaj imię i nazwisko osoby dla której chcesz otrzymac rekomendaje (zgodnie z danymi z pliku JSON): ")

    # read data json file
    with open(rating_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    movie_suggestor(data, user)
