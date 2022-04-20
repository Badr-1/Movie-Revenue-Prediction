import requests
import string
import pandas as pd

API_KEY = "95a1eb7e26b4cc39736c3f8dd4d719b0"
data = pd.read_csv('merged_w_directors.csv')


def get_movie(movie_name):
    movie_name = movie_name.translate(str.maketrans('', '', string.punctuation)).replace(' ', '+')
    response = requests.get(
        'https://api.themoviedb.org/3/search/movie?api_key={0}&query={1}'.format(API_KEY, movie_name))
    if (response.status_code == 200):
        # returns id of the first result as it is most likely to be the movie we're looking for
        if (len(response.json()["results"]) > 0):
            return response.json()["results"][0]['id']
        else:
            print("Movie name: {0}".format(movie_name))
    return


def get_director(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{0}/credits?api_key={1}'.format(movie_id, API_KEY))
    if (response.status_code == 200):
        for member in response.json()["crew"]:
            if member["job"] == "Director":
                return member["name"]
    return


def fill_df():
    x = data.loc[data['director'].isna()]
    for index in x.index:
        movie_id = get_movie(data['movie_title'].iloc[index])
        director = get_director(movie_id)
        data['director'].iloc[index] = director


data.to_csv("merged_w_directors.csv", index=False, na_rep="NA")
# print(data['director'].isna().sum())
