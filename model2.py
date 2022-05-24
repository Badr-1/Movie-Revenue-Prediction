import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn import svm, tree
import pickle 
from sklearn.metrics import precision_recall_fscore_support

# Merge the three csv files into a single file
def merge_data():
    act_data = pd.read_csv("data2/movie-voice-actors.csv")
    dir_data = pd.read_csv("data2/movie-director.csv")
    class_data = pd.read_csv("data2/movies-revenue-classification.csv")

    # rename column titles for merging
    act_data.rename(columns={"movie": "movie_title"}, inplace=True)
    dir_data.rename(columns={"name": "movie_title"}, inplace=True)

    # merge all 3 tables using movie_title key
    merged_data = pd.merge(class_data, dir_data, on="movie_title", how="outer")
    merged_data = pd.merge(merged_data, act_data, on="movie_title", how="outer")

    return merged_data

# fill null values
def fill_actors(data):
    data["voice-actor"].fillna(0,inplace=True)

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
    data["director"] = data["director"].apply(lambda x: 1 if x in top_directors else 0)

def encode_actors(data):
    """
    acts_freq = list(data["voice-actor"].value_counts().to_dict().keys())[0:1]
    data["voice-actor"] = data["voice-actor"].apply(lambda x: 1 if x in acts_freq else 0)
    """

    acts_freq = data["voice-actor"].value_counts().to_dict()
    data["voice-actor"] = data["voice-actor"].apply(lambda x: acts_freq[x]/len(data))

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
    data['release_date'] = data['release_date'].apply(lambda x: x + 1900 if x > 23 and x <= 99 else x + 2000)

    data["new_movie"] = 0
    for i in range(len(data)):
        if data['release_date'].iloc[i] >= 1999:
            data.iloc[i, data.columns.get_loc('new_movie')] = 1

# Normalize data
def normalize_data(data):
    for col in data:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

# Train model
def train_model(x_train, y_train, x_test, y_test, choice):
    model = None
    if choice == 'svm':
        model = svm.SVC(kernel='rbf', gamma=0.8, C=2.0)
        choice = 'SVM'

    elif choice == 'tree':
        model = tree.DecisionTreeClassifier(max_depth=2)
        choice = 'Decision Tree'

    elif choice == 'adaboost':
        model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=200)
        choice = 'Adaboost'

    elif choice == 'logistic':
        model = LogisticRegression(solver="lbfgs", random_state=612)
        choice = 'Logistic Regression'

    else:
        print("Unknown model")
        return

    model.fit(x_train,y_train)
    y_prediction = model.predict(x_train)
    accuracy = np.mean(y_prediction == y_train)
    precision, recall, fscore, support = precision_recall_fscore_support(y_train, y_prediction, zero_division=1)
    print("Train {0} accuracy:".format(choice), (accuracy * 100))
    print("Train {0} precision: {1}".format(choice, precision))
    print("Train {0} recall: {1}".format(choice, recall))
    print("Train {0} fscore: {1}".format(choice, fscore))
    print("Train {0} support: {1}".format(choice, support))
    print()
    y_prediction = model.predict(x_test)
    accuracy=np.mean(y_prediction == y_test)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, y_prediction,zero_division=1)
    print("Test {0} accuracy:".format(choice), (accuracy * 100))
    print("Test {0} precision: {1}".format(choice,precision))
    print("Test {0} recall: {1}".format(choice,recall))
    print("Test {0} fscore: {1}".format(choice,fscore))
    print("Test {0} support: {1}".format(choice,support))
    print()

    filename = ('Models/{0}Model.sav'.format(choice))
    pickle.dump(model, open(filename, 'wb'))

# merge data
data = merge_data()

# drop null labels
# 353 missing labels
data.dropna(subset=['MovieSuccessLevel'], inplace=True)

# TODO need justification
data.drop(["character","movie_title"], inplace=True, axis=1)

# split data
y = data["MovieSuccessLevel"]
x = data.drop("MovieSuccessLevel", axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9, test_size=0.2)

# ----------- Preprocess train data -----------

# Fill null values
fill_actors(x_train)
fill_directors(x_train)
fill_genres(x_train)
fill_ratings(x_train)

# Encode features
encode_label(y_train)
encode_directors(x_train)
encode_actors(x_train)
encode_genre(x_train)
encode_ratings(x_train)
release_date_feature_extraction(x_train)

# Normalize
normalize_data(x_train)

# ----------- Preprocess test data -----------

# Fill null values
fill_actors(x_test)
fill_directors(x_test)
fill_genres(x_test)
fill_ratings(x_test)

# Encode features
encode_label(y_test)

encode_directors(x_test)
encode_actors(x_test)
encode_genre(x_test)
encode_ratings(x_test)
release_date_feature_extraction(x_test)

normalize_data(x_test)

# ----------- Model training -----------

train_model(x_train, y_train, x_test, y_test, "svm")
train_model(x_train, y_train, x_test, y_test, "tree")
train_model(x_train, y_train, x_test, y_test, "adaboost")
train_model(x_train, y_train, x_test, y_test, "logistic")
