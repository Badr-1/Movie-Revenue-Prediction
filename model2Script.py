import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

def loadModels():
    filename = "Models/AdaboostModel.sav"
    Adaboost = pickle.load(open(filename, 'rb'))

    filename = "Models/Logistic RegressionModel.sav"
    LogisticReg = pickle.load(open(filename, 'rb'))

    filename = "Models/Decision TreeModel.sav"
    DecisionTree = pickle.load(open(filename, 'rb'))

    filename = "Models/SVMModel.sav"
    svm = pickle.load(open(filename, 'rb'))

    return svm, DecisionTree, Adaboost, LogisticReg


def load_data():
    act_data = pd.read_csv(
        r"ProjectTestSamples\Milestone 2\movies\movie-voice-actors-test-samples.csv"
    )
    dir_data = pd.read_csv(
        r"ProjectTestSamples\Milestone 2\movies\movie-director-test-samples.csv"
    )
    class_data = pd.read_csv(
        r"ProjectTestSamples\Milestone 2\movies\movies-revenue-test-samples.csv"
    )

    # rename column titles for merging
    act_data.rename(columns={"movie": "movie_title"}, inplace=True)
    dir_data.rename(columns={"name": "movie_title"}, inplace=True)

    # merge all 3 tables using movie_title key
    merged_data = pd.merge(class_data, dir_data, on="movie_title", how="outer")
    merged_data = pd.merge(merged_data,
                           act_data,
                           on="movie_title",
                           how="outer")

    return merged_data


# fill null values
def fill_actors(data):
    data["voice-actor"].fillna(0, inplace=True)


def fill_directors(data):
    s = data["director"].value_counts()
    data["director"].fillna(s.index[0], inplace=True)


def fill_genres(data):
    e = data["genre"].value_counts()
    data["genre"].fillna(e.index[len(e) // 2], inplace=True)


def fill_ratings(data):
    x = data["MPAA_rating"].value_counts()
    data["MPAA_rating"].fillna(x.index[len(x) // 2], inplace=True)


# encode features
def encode_label(data):
    """
    enc_ord = OrdinalEncoder()
    data = enc_ord.fit_transform([data])
    """
    """
    lbl_enc = LabelEncoder()
    data = lbl_enc.fit_transform(data)
    """
    success_levels = sorted(data.unique())
    success_levels.insert(0, success_levels.pop())
    success_levels.reverse()
    data = data.apply(lambda x: success_levels.index(x))


def encode_directors(data):
    # Done tweaking
    top_directors = list(data["director"].value_counts().to_dict().keys())[1:]
    data["director"] = data["director"].apply(lambda x: 1
                                              if x in top_directors else 0)


def encode_actors(data):
    """
    acts_freq = list(data["voice-actor"].value_counts().to_dict().keys())[0:1]
    data["voice-actor"] = data["voice-actor"].apply(lambda x: 1 if x in acts_freq else 0)
    """

    acts_freq = data["voice-actor"].value_counts().to_dict()
    data["voice-actor"] = data["voice-actor"].apply(
        lambda x: acts_freq[x] / len(data))


def encode_genre(data):
    enc_ord = OrdinalEncoder()
    data["genre"] = enc_ord.fit_transform(data[["genre"]])


def encode_ratings(data):
    enc_ord = OrdinalEncoder()
    data["MPAA_rating"] = enc_ord.fit_transform(data[["MPAA_rating"]])


def release_date_feature_extraction(data):
    data['day'] = pd.DatetimeIndex(data['release_date']).day
    data['month'] = pd.DatetimeIndex(data['release_date']).month

    data['release_date'] = data['release_date'].str[-2:].astype(int)
    data['release_date'] = data['release_date'].apply(
        lambda x: x + 1900 if x > 23 and x <= 99 else x + 2000)

    data["new_movie"] = 0
    for i in range(len(data)):
        if data['release_date'].iloc[i] >= 1999:
            data.iloc[i, data.columns.get_loc('new_movie')] = 1


# Normalize data
def normalize_data(data):
    for col in data:
        data[col] = (data[col] - data[col].mean()) / data[col].std()


svm, DecisionTree, Adaboost, LogisticReg = loadModels()
data = load_data()
data.dropna(subset=['MovieSuccessLevel'], inplace=True)
data.drop(['movie_title', "character"], inplace=True, axis=1)

data = data.sample(frac=1)
data = data.iloc[0:30, :]

x_test = data.drop("MovieSuccessLevel", axis=1)
y_test = data["MovieSuccessLevel"]

encode_label(y_test)

fill_actors(x_test)
fill_directors(x_test)
fill_genres(x_test)
fill_ratings(x_test)

encode_directors(x_test)
encode_actors(x_test)
encode_ratings(x_test)
encode_genre(x_test)
release_date_feature_extraction(x_test)

# normalize_data(x_test)
scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)

def printMetrics(model, x_test, y_test):
    model.predict(x_test)
    y_prediction = model.predict(x_test)
    accuracy = np.mean(y_prediction == y_test)
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test, y_prediction, zero_division=1)
    print("Accuracy:", accuracy * 100)
    print("precision: {0}".format(precision))
    print("recall: {0}".format(recall))
    print("fscore: {0}".format(fscore))
    print("support: {0}".format(support))
    print()



print("SVM:")
printMetrics(svm, x_test, y_test)

print("Decision Tree:")
printMetrics(DecisionTree, x_test, y_test)

print("Adaboost:")
printMetrics(Adaboost, x_test, y_test)

print("Logistic Regression:", )
printMetrics(LogisticReg, x_test, y_test)
