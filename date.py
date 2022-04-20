import pandas as pd



def format_date():
    data = pd.read_csv('movies-revenue.csv')
    data['day'] = pd.DatetimeIndex(data['release_date']).day
    data['month'] = pd.DatetimeIndex(data['release_date']).month
    data['weekend'] = 0
    data['Summer'] = 0
    data['Spring'] = 0
    data['Autumn'] = 0
    data['Winter'] = 0
    data['release_date'] = data['release_date'].str[-2:].astype(int)
    data['revenue'] = data['revenue'].str[1:].replace(',', '', regex=True).astype('int64')
    # Getting the year in which the film was released
    for i in range(len(data)):
        if data.at[i, 'release_date'] > 23 and data.at[i,'release_date'] <= 99:
            data.at[i, 'release_date'] = 1900 + data.at[i, 'release_date']
        else:
            data.at[i, 'release_date'] = 2000 + data.at[i, 'release_date']
    # Getting the season of the year
    for i in range(len(data)):
        if data.at[i,'month'] >= 3 and data.at[i, 'month'] <= 5:
            data.at[i, 'Spring'] = 1
        elif data.at[i,'month'] >= 6 and data.at[i, 'month'] <= 8:
            data.at[i, 'Summer'] = 1
        elif data.at[i,'month'] >= 9 and data.at[i, 'month'] <= 11:
            data.at[i, 'Autumn'] = 1
        elif data.at[i,'month'] >= 12 and data.at[i, 'month'] <= 2:
            data.at[i, 'Winter'] = 1
    # Checking if the released date was a weekend or not
    for i in range(len(data)):
        data.at[i, 'weekend'] = pd.Timestamp(data.at[i, 'release_date'], data.at[i, 'month'], data.at[i, 'day'])
        if data.at[i, 'weekend'].weekday() >= 5:
            data.at[i, 'weekend'] = 1
        else:
            data.at[i, 'weekend'] = 0

    data['release_date'] = data['release_date'] * 100 + data['month']
    print(data[['Spring', 'Summer', 'Autumn', 'Winter', 'release_date', 'weekend']])



