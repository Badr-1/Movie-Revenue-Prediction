import datetime

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import requests
import string

# ============================================================
API_KEY = "95a1eb7e26b4cc39736c3f8dd4d719b0"


# data = pd.read_csv('merged_w_directors.csv')
def get_genre(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{0}?api_key={1}'.format(movie_id, API_KEY))
    if (response.status_code == 200):
        return response.json()['genres'][0]['name']
    return


def get_revenue(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{0}?api_key={1}'.format(movie_id, API_KEY))
    if (response.status_code == 200):
        return response.json()['revenue']
    return


def get_rating(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{0}?api_key={1}&language=en-US&append_to_response=release_dates'.format(
            movie_id, API_KEY))
    if (response.status_code == 200):
        for release in response.json()['release_dates']['results']:
            if (release['iso_3166_1'] == 'US'):
                for date in release['release_dates']:
                    if (date['certification'] != ''):
                        return date['certification']
    return


def get_release_date(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{0}?api_key={1}'.format(movie_id, API_KEY))
    if (response.status_code == 200):
        date = response.json()['release_date']
        formatted_date = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%d-%b-%y')
        return formatted_date
    return


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


def fill_directors():
    x = data.loc[data['director'].isna()]
    for index in x.index:
        movie_id = get_movie(data['movie_title'].iloc[index])
        director = get_director(movie_id)
        data['director'].iloc[index] = director


def fill_genre():
    x = data.loc[data['genre'].isna()]
    for index in x.index:
        movie_id = get_movie(data['movie_title'].iloc[index])
        genre = get_genre(movie_id)
        data['genre'].iloc[index] = genre


def fill_rating():
    x = data.loc[data['MPAA_rating'].isna()]
    for index in x.index:
        movie_id = get_movie(data['movie_title'].iloc[index])
        rating = get_rating(movie_id)
        data['MPAA_rating'].iloc[index] = rating


def fill_release_date():
    x = data.loc[data['release_date'].isna()]
    for index in x.index:
        movie_id = get_movie(data['movie_title'].iloc[index])
        release_date = get_release_date(movie_id)
        data['release_date'].iloc[index] = release_date


def fill_revenue():
    x = data.loc[data['revenue'].isna()]
    for index in x.index:
        movie_id = get_movie(data['movie_title'].iloc[index])
        revenue = get_revenue(movie_id)
        data['revenue'].iloc[index] = revenue


# Label encoding for MPAA_rate
def rating_label_encoding():
    lbl_encode = LabelEncoder()
    data['rate'] = lbl_encode.fit_transform(data['MPAA_rating'])
    # print(data['rate'])
    # print(data['MPAA_rating'])
    #
    # # print("data with out not rated:\n", data_with_notRated)
    # print("MPAA rate Unique values: \n", data['MPAA_rating'].unique())
    # print("rate Unique values: \n", data['rate'].unique())

    # plotting relation between rate after label encoding and revenue
    # plt.scatter(data['rate'], data['revenue'])
    # plt.xlabel('Rate (lbl encode', fontsize=12)
    # plt.ylabel('revenue', fontsize=12)
    # plt.show()
    data.drop(['MPAA_rating'], axis=1, inplace=True)


# Ordinal encoder
def rating_ordinal_encoding():
    ord_encode = OrdinalEncoder()
    # data to Series
    sr = pd.Series(data['MPAA_rating'])
    # Series to 2d array
    # rate_array = sr.as_matrix().reshape(1, len(data['MPAA_rating']))
    # ord_result = ord_encode.fit_transform(rate_array)
    # print(data['rate'])
    # print(data['MPAA_rating'])
    mapped_rate = {'R': 1, 'PG': 2, 'PG-13': 3, 'G': 4, 'Not Rated': 3, 'NR': 3}
    data['rate_ordinal'] = data['MPAA_rating'].replace(mapped_rate)
    # print("Ordinal\n", data['rate_ordinal'])
    # print("Original\n", data['MPAA_rating'])

    data.drop(['MPAA_rating'], axis=1, inplace=True)

    plt.scatter(data['rate_ordinal'], data['revenue'])
    plt.xlabel('Rate (lbl encode)', fontsize=12)
    plt.ylabel('revenue', fontsize=12)
    # plt.show()


# One-Hot encoder
def rating_one_hot_encoding():
    global data
    rate_one_hot = pd.get_dummies(data['MPAA_rating'])
    rate_one_hot.drop(['Not Rated'], axis=1, inplace=True)
    data = pd.concat([data, rate_one_hot], axis=1)

    # print(rate_one_hot)
    data_len = len(data['revenue'])
    # data['revenue'] = data['revenue'].astype(str).apply(lambda x: x.replace('-', '')).astype(int)
    # print("revenue")
    # print(data['revenue'])
    data.drop(['MPAA_rating'], axis=1, inplace=True)


def genre_one_hot_encoding():
    global data
    genre_one_hot = pd.get_dummies(data['genre'])
    data = pd.concat([data, genre_one_hot], axis=1)
    data.drop(['genre'], axis=1, inplace=True)


def release_date_feature_extraction():
    data = pd.read_csv('merged_full_datavTest.csv')
    data['day'] = pd.DatetimeIndex(data['release_date']).day
    data['month'] = pd.DatetimeIndex(data['release_date']).month
    data['weekend'] = 0
    # data['Summer'] = 0
    # data['Spring'] = 0
    # data['Autumn'] = 0
    # data['Winter'] = 0
    data['new_movie'] = 0
    data['release_date'] = data['release_date'].str[-2:].astype(int)
    # data['revenue'] = data['revenue'].str[1:].replace(',', '', regex=True).astype('int64')
    # Getting the year in which the film was released
    for i in range(len(data)):
        if data['release_date'].loc[i] > 23 and data['release_date'].loc[i] <= 99:
            data.at[i, 'release_date'] = 1900 + data.at[i, 'release_date']
        else:
            data.at[i, 'release_date'] = 2000 + data.at[i, 'release_date']
    # Getting the season of the year
    # for i in range(len(data)):
    #     if data.at[i, 'month'] in [3, 4, 5]:
    #         data.at[i, 'Spring'] = 1
    #     elif data.at[i, 'month'] in [6, 7, 8]:
    #         data.at[i, 'Summer'] = 1
    #     elif data.at[i, 'month'] in [9, 10, 11]:
    #         data.at[i, 'Autumn'] = 1
    #     elif data.at[i, 'month'] in [1, 2, 12]:
    #         data.at[i, 'Winter'] = 1
    # column for new movies (from 2005)
    for i in range(len(data)):
        if data.at[i, 'release_date'] >= 2005:
            data.at[i, 'new_movie'] = 1

    # Checking if the released date was a weekend or not
    for i in range(len(data)):
        data.at[i, 'weekend'] = pd.Timestamp(data.at[i, 'release_date'], data.at[i, 'month'], data.at[i, 'day'])
        if data.at[i, 'weekend'].weekday() >= 5:
            data.at[i, 'weekend'] = 1
        else:
            data.at[i, 'weekend'] = 0

    # data['release_date'] = data['release_date'] * 100 + data['month']
    # print(data[['Spring', 'Summer', 'Autumn', 'Winter', 'release_date', 'weekend']])
    return data


def merge_data():
    act_data = pd.read_csv("movie-voice-actors.csv")
    dir_data = pd.read_csv("movie-director.csv")
    rev_data = pd.read_csv("movies-revenue.csv")

    # rename column titles for merging
    act_data.rename(columns={"movie": "movie_title"}, inplace=True)
    dir_data.rename(columns={"name": "movie_title"}, inplace=True)

    # merge all 3 tables using movie_title key
    merged_data = pd.merge(rev_data, dir_data, on="movie_title", how="outer")
    merged_data = pd.merge(merged_data, act_data, on="movie_title", how="outer")

    # reorder revenue column
    merged_data = merged_data[[col for col in merged_data.columns if col != "revenue"] + ["revenue"]]
    return merged_data


def directors_encoding():
    directors_movies = data['director'].value_counts()
    for x in directors_movies.index:
        if directors_movies[x] >= 3:
            data[x] = 0
            data.loc[data.director == x, x] = 1

    data.drop(['director'], axis=1, inplace=True)

# def director_target_encoding():



def preprocessing():
    # removing $ and ,
    data["revenue"] = data["revenue"].str.replace('$', '', regex=False)
    data["revenue"] = data["revenue"].str.replace(',', '', regex=False)
    data["revenue"] = data["revenue"].astype(float)
    # data['revenue'].fillna(value=0, inplace=True)

    # print((data["revenue"] % 1 == 0).all())
    # todo try with and without
    # data.loc[data['revenue'] == 0, 'revenue'] = data['revenue'].mean()
    # TODO waiting for dates
    # format_date()

    # encoding
    # feature selection


# ==============================================================

# data = merge_data()
# fill_revenue()
# fill_release_date()
# fill_genre()
# fill_rating()
# fill_directors()
# preprocessing()
# data.to_csv('merged_full_data.csv',index = False)


global data
data = pd.read_csv("merged_full_data.csv")

data.drop(['character', 'voice-actor', 'movie_title'], axis=1, inplace=True)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True, subset=['revenue', 'director', 'MPAA_rating'])

data = data[data.revenue != 0]

# rating_label_encoding()
# rating_one_hot_encoding()
rating_ordinal_encoding()

genre_one_hot_encoding()
data.to_csv('merged_full_datavTest.csv', index=False)
data = release_date_feature_extraction()
directors_encoding()
data = data[[col for col in data.columns if col != "revenue"] + ["revenue"]]

y = data['revenue']
x = data.iloc[:,0:-2]


# scaling
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, shuffle=True, random_state=45)


cls = linear_model.LinearRegression()
cls.fit(x_train, y_train)
prediction_linear = cls.predict(x_test)

print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction_linear))
data.to_csv('merged_full_datavTest.csv', index=False)





# poly_features = PolynomialFeatures(degree=3)
# x_train_poly = poly_features.fit_transform(x_train)
#
# poly_model = linear_model.LinearRegression()
# poly_model.fit(x_train_poly, y_train)
#
#
# y_train_predicted = poly_model.predict(x_train_poly)
# ypred = poly_model.predict(poly_features.transform(x_test))
#
# # predicting on test data-set
# prediction_ploy = poly_model.predict(poly_features.fit_transform(x_test))
#
# print('Mean Square Error', metrics.mean_squared_error(y_test, prediction_ploy))
#
# true_film_revenue = np.asarray(y_test)[0]
# predicted_film_revenue = prediction_ploy[0]
# print('True revenue for the first film in the test set in millions is : ' + str(true_film_revenue))
# print('Predicted revenue for the first film in the test set in millions is : ' + str(predicted_film_revenue))

