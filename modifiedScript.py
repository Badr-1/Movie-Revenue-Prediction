# %% [markdown]
# steps:
# 1. merge_data based on movie title
# 2. clean data remove $ and , from revenue
# 3. encoding
# 4. split data into train and test
# 5. train model
# 6. predict
# 7. evaluate
# 8. save model

# %% [markdown]
# # Load, clean, and fill data

# %%

import datetime
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import pickle
import joblib
import csv
import scipy.stats as stats


def merge_data():
    act_data = pd.read_csv("movie-voice-actors.csv")
    dir_data = pd.read_csv("movie-director.csv")
    rev_data = pd.read_csv("movies-revenue.csv")
    act_data.rename(columns={"movie": "movie_title"}, inplace=True)
    dir_data.rename(columns={"name": "movie_title"}, inplace=True)
    merged_data = pd.merge(rev_data, dir_data, on="movie_title", how="outer")
    merged_data = pd.merge(merged_data,
                           act_data,
                           on="movie_title",
                           how="outer")
    merged_data = merged_data[
        [col for col in merged_data.columns if col != "revenue"] + ["revenue"]]
    return merged_data


def cleaning(data):
    # removing $ and ,
    data["revenue"] = data["revenue"].str.replace('$', '', regex=False)
    data["revenue"] = data["revenue"].str.replace(',', '', regex=False)
    data["revenue"] = data["revenue"].astype(float)
    return data


def fill(data):
    data = data.fillna(axis=0, method='ffill')
    data = data.fillna(axis=0, method='backfill')
    return data



def remove_outlier(data):
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]
    return data





def encoding(data):
    df = data.copy()
    print(df.columns)
    # rating
    df = rating_ordinal_encoding(df)
    # release date
    # df = release_date_feature_extraction(df)
    df = mean_encoding_of(df, 'release_date')
    # genre
    df = label_encoding_of(df, 'genre')
    # df = label_encoding_of(df, 'genre')
    # movie title
    df = label_encoding_of(df, 'movie_title')
    # voice actor
    df = label_encoding_of(df, 'voice-actor')
    # character
    df = label_encoding_of(df, 'character')
    # director
    df = mean_encoding_of(df, 'director')
    return df




def mean_encoding_of(data, col):
    encoder = TargetEncoder()
    data[col] = encoder.fit_transform(data[col], data['revenue'])
    return data


def label_encoding_of(data, col):
    lbl_encode = LabelEncoder()
    data[col] = lbl_encode.fit_transform(data[col])
    return data



def rating_ordinal_encoding(data):
    ord_encode = OrdinalEncoder()
    sr = pd.Series(data['MPAA_rating'])
    mapped_rate = {
        'R': 1,
        'PG': 2,
        'PG-13': 3,
        'G': 4,
        'Not Rated': 3,
        'NR': 3
    }
    # mapped_rate = {'Not Rated': 0, 'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4}
    data['MPAA_rating'] = data['MPAA_rating'].replace(mapped_rate)
    return data




def release_date_feature_extraction(data):
    # data['day'] = pd.DatetimeIndex(data['release_date']).day
    data['month'] = pd.DatetimeIndex(data['release_date']).month
    # data['new_movie'] = 0
    data['year'] = data['release_date'].str[-2:].astype(int)
    for i in range(len(data)):
        x = data['year'].iloc[i]
        if x > 23 and x <= 99:
            data.iloc[i, data.columns.get_loc('year')] += 1900
        else:
            data.iloc[i, data.columns.get_loc('year')] += 2000

    # for i in range(len(data)):
    #     if data['year'].iloc[i] >= 2005:
    #         data.iloc[i, data.columns.get_loc('new_movie')] =1
    data.drop(['release_date'], axis=1, inplace=True)
    return data


def directors_modified_one_hot_encoding(data):
    directors_movies = data['director'].value_counts()
    for x in directors_movies.index:
        if directors_movies[x] >= 3:
            data[x] = 0
            data.loc[data.director == x, x] = 1
    data.drop(['director'], axis=1, inplace=True)
    return data




def feature_selection(data):
    corr = data.corr()
    # Top 21% Correlation training features with the Value
    top_feature = corr.index[abs(corr['revenue']) > 0.21]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    top_feature = top_feature.drop(['revenue'])
    # plt.show()
    return top_feature






def divide_data(data,top_feature):
    y = data['revenue']
    print(top_feature)
    x = data[top_feature]
    return x, y



def apply_scalling(x):
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    return x



def split_to_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test



data = merge_data()
data = cleaning(data)
data = fill(data)
data = encoding(data)
x, y = divide_data(data,feature_selection(data))
x = apply_scalling(x)

# load and predict
loaded_poly_features = joblib.load('poly_features')
loaded_poly_model = joblib.load('poly_model')
loaded_ridg_model = joblib.load('ridg_model')

X_val_prep = loaded_poly_features.transform(x)
poly_pred = loaded_poly_model.predict(X_val_prep)
ridg_pred = loaded_ridg_model.predict(x)

print("")
prediction_ridge = loaded_ridg_model.predict(x)
print('MSE ridge {All}',
      metrics.mean_squared_error(np.asarray(y), prediction_ridge))
print("R2 Score ridge", metrics.r2_score(np.asarray(y), prediction_ridge))
print("")

print("")
prediction_ploy = loaded_poly_model.predict(X_val_prep)
print('MSE poly {All}',
      metrics.mean_squared_error(np.asarray(y), prediction_ploy))
print("R2 Score poly", metrics.r2_score(np.asarray(y), prediction_ploy))
print("")
