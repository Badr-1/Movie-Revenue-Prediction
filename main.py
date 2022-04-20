import pandas as pd

def main():
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
    
    # remove $ and ,
    merged_data["revenue"] = merged_data["revenue"].str.replace('$','',regex=False)
    merged_data["revenue"] = merged_data["revenue"].str.replace(',','')
   
    # unnecessary, revenue doesnt contain floats
    # print((merged_data["revenue"] % 1  == 0).all()) > True
    # merged_data["revenue"] = merged_data["revenue"].astype(float)

    # remove NA's
    # print("size BEFORE dropping NA's:",merged_data.shape)
    merged_data.dropna(inplace=True)
    # print("size AFTER dropping NA's:",merged_data.dropna().shape)
    
    # too many unique directors, characters and voice actors, make one-hot encoding impossible
    for f in merged_data:
        print(f,len(merged_data[f].unique()),'\n')

    # print(merged_data["MPAA_rating"].unique())
    # print(merged_data["genre"].unique())

    # print(merged_data.groupby("genre")['revenue'].mean())

    # export new merged data as csv
    merged_data.to_csv("merged.csv", index=False,na_rep="NA",)
    # export new merged data as aligned text
    with open("merged.txt", "w") as f:
        f.write(merged_data.__repr__())


if __name__ == "__main__":
    main()
