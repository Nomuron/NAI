"""
Authors: Magdalena Asmus-Mrzygłów, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
numpy package
json package
Pycharm or other IDE for Python
Link to install python: https://www.python.org/downloads/
To run script you need to run it from IDE .
You will be asked to enter the details of person for whom you are looking for recommendations.

==========================================
Recommendations algorithms
==========================================

There are few recommendation engines/algorithms available to resolve recommendation problem.
Two most popular are:
1. Euclidean distance score
Where this value=0 it means that two vectors are axactky the same.
2.Pearson correlation score
A number between –1 and 1 that measures the strength and direction of the relationship between two variables.

In this implementation we are using Euclidean distance score.

The purpose of the script is to suggest 5 movies recommended for viewing and 5 movies not recommended
for a specified person.
Algorythm receive JSON file with movie rating provided by individuals.
On this basis euclidean distance score is calculated.

We decided to use person with the smallest euclidean distance as a source of recommendations.
5 movies from his/her list with the biggest ratings are listed as a recommendation
and 5 movies with the smallest ratings as discouraged.
"""
import json

import numpy as np


def euclidean_score(dataset, user1, user2):
    """Calculates euclidian distance value for two individuals

     Parameters:
     dataset (dictionary): data from JSON,
     user1, user2 (string): users for whome we calculate the euclidian distance value,

     Returns:
     [float]: euclidian distance value
     """
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
    """Provides recommendation for user1. 5 recommended movies and 5 movies to avoid.

     Parameters:
     dataset (dictionary): data from JSON,
     user1 (String): user for whom we are calculating recommendations.

     Prints:
     List[str]: List of recommended movies,
     List[str]: List of movies to avoid.
     """
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
