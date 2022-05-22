import datetime
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pickle

def loadModels():
    filename = "linearRidgModel.sav"
    model1 = pickle.load(open(filename, 'rb'))
    filename = "polyRegModel.sav"
    model2 = pickle.load(open(filename, 'rb'))
    return model1,model2
def loadTest():
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

    # data = merge_data()
    data = pd.read_csv("merged_full_data.csv")
    ytest = data["revenue"]
    xtest = data.drop(["revenue"],axis=1)
    return xtest,ytest
def release_date_feature_extraction(data):
    data['day'] = pd.DatetimeIndex(data['release_date']).day
    data['month'] = pd.DatetimeIndex(data['release_date']).month
    data['new_movie'] = 0
    data['release_date'] = data['release_date'].str[-2:].astype(int)
    # Getting the year in which the film was released
    for i in range(len(data)):
        x = data['release_date'].iloc[i]
        if x > 23 and x <= 99:
            data.iloc[i, data.columns.get_loc('release_date')] += 1900
        else:
            data.iloc[i, data.columns.get_loc('release_date')] += 2000

    # column for new movies (from 2005)
    for i in range(len(data)):
        if data['release_date'].iloc[i] >= 2005:
            data.iloc[i, data.columns.get_loc('new_movie')] =1
def rating_label_encoding(data):
    lbl_encode = LabelEncoder()
    data['rate'] = lbl_encode.fit_transform(data['rate'])
    data.drop(['rate'], axis=1, inplace=True)
def genre_ordinal_encoding(data):
    mapped_genre = {'Comedy': 9, 'Adventure': 13, 'Drama': 7, 'Action': 12, 'Musical': 15, 'Romantic Comedy': 8,
                    'Horror': 3, \
                    'Thriller/Suspense': 6, 'Crime': 1, 'Documentary': 2, 'Fantasy': 14, 'Black Comedy': 4,
                    'Western': 5, \
                    'Romance': 10, 'Animation': 11}
    data['genre'] = data['genre'].replace(mapped_genre)
def preprocess(xtest,ytest):
    # Y processing
    # removing $ and ,
    # ytest = ytest.str.replace('$', '', regex=False)
    # ytest = ytest.str.replace(',', '', regex=False)
    # ytest = ytest.astype(float)
    # ytest = ytest.fillna(ytest.mean())

    # X processing
    # load and add pretrained feature columns
    with open ("features.csv") as f:
        features = f.read().split(",")
        features = [x.strip('\n')for x in features]
        for feature in features:
            if not feature in xtest.columns:
                xtest[feature] = 0

        xtest.loc[xtest.director == "David Hand", "David Hand"] = 1
        xtest = xtest[xtest.columns.intersection(features)]

    rating_label_encoding(xtest)
    genre_ordinal_encoding(xtest)
    release_date_feature_extraction(xtest)
    return xtest,ytest

linearModel, polyModel = loadModels()
xtest,ytest = loadTest()
xtest.dropna(inplace=True, subset=['release_date'])
xtest, ytest = preprocess(xtest,ytest)
# TODO: Issue genre still contains string ie family
print(linearModel.score(xtest,ytest))
print(polyModel.score(xtest,ytest))