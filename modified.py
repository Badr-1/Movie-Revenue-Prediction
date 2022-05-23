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
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import scipy.stats as stats
import csv

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


data = merge_data()
data = cleaning(data)
data = fill(data)

df = data.copy()
print(df.columns)
# rating
df = rating_ordinal_encoding(df)
# release date
df = mean_encoding_of(df, 'release_date')
# genre
df = label_encoding_of(df, 'genre')
# movie title
df = label_encoding_of(df, 'movie_title')
# voice actor
df = label_encoding_of(df, 'voice-actor')
# character
df = label_encoding_of(df, 'character')
# director
df = mean_encoding_of(df, 'director')

dataX = df.drop(['revenue'], axis=1)
cor_matrix = dataX.corr().abs()
upper_tri = cor_matrix.where(
    np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
to_drop = [
    column for column in upper_tri.columns if any(upper_tri[column] > 0.7)
]
print(to_drop)
df = df.drop(to_drop, axis=1)

corr = df.corr()
top_feature = corr.index[abs(corr['revenue']) > 0.21]
plt.subplots(figsize=(12, 8))
top_corr = df[top_feature].corr()
sns.heatmap(top_corr, annot=True)
top_feature = top_feature.drop(['revenue'])

y = df['revenue']
print(top_feature)
x = df[top_feature]

with open ("featurez.csv","w") as f:
    l = [x.strip('\n') for x in top_feature]
    csv.writer(f).writerow(l)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# splitting data to train and test
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=.20,
                                                    shuffle=True,
                                                    random_state=45)

#Ridge regression
ridg = linear_model.Ridge()
ridg.fit(x_train, y_train)

print("")
prediction_ridge_train = ridg.predict(x_train)
print('MSE ridge {Train}',
      metrics.mean_squared_error(np.asarray(y_train), prediction_ridge_train))
prediction_ridge = ridg.predict(x_test)
print('MSE ridge {Test}',
      metrics.mean_squared_error(np.asarray(y_test), prediction_ridge))
print("R2 Score ridge", metrics.r2_score(np.asarray(y_test), prediction_ridge))
print("")

true_film_value = np.asarray(y_test)[0]
predicted_film_value = prediction_ridge[0]
print('True value for the first film in the test set is : ' +
      str(true_film_value))
print('Predicted value for the first film in the test set is : ' +
      str(predicted_film_value))
print("Difference between true and predicted value is : " +
      str(true_film_value - predicted_film_value))

# Polynomial regression
poly_features = PolynomialFeatures(degree=3)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.fit_transform(x_test)

poly_model = linear_model.LinearRegression()
poly_model.fit(x_train_poly, y_train)

print("")
prediction_ploy_train = poly_model.predict(x_train_poly)
print('MSE Poly {TRAIN}',
      metrics.mean_squared_error(y_train, prediction_ploy_train))
prediction_ploy_test = poly_model.predict(x_test_poly)
print('MSE Poly {TEST}',
      metrics.mean_squared_error(y_test, prediction_ploy_test))
print("R2 Score Poly", metrics.r2_score(y_test, prediction_ploy_test))
print("")

true_film_value = np.asarray(y_test)[0]
predicted_film_value = prediction_ploy_test[0]
print('True value for the first film in the test set is : ' +
      str(true_film_value))
print('Predicted value for the first film in the test set is : ' +
      str(predicted_film_value))
print("Difference between true and predicted value is : " +
      str(true_film_value - predicted_film_value))

# saving models
joblib.dump(ridg, 'ridg_model')
joblib.dump(poly_model, 'poly_model')
joblib.dump(poly_features, 'poly_features')
