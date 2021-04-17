import pandas as pd

# this script cleans raw data, extracts features, and moves sentiment label to last column of df
if __name__ == "__main__":
    # data cleaning
    header = ['text', 'id', 'place', 'created_at', 'user_location', 'user_name', 'followers_count', 'retweet_count',
              'favorite_count', 'hashtags', 'sentiment']
    # files = ['train.csv','dev.csv','test.csv']
    directory = 'train-dev-test/'
    files = ['train_mini.csv']
    for file in files:
        print('cleaning and extracting features from', file)
        df = pd.DataFrame(columns=header)
        df = pd.read_csv(directory+file,header=0,low_memory=False,encoding='utf-8-sig')
        df[['weekday', 'month','day','time','useless','year']] = df.created_at.str.split(expand=True)
        df.drop(columns=['id', 'user_name', 'created_at', 'useless'],inplace=True)

        # presence of numbers 
        df['num_present'] = df['text'].map(lambda x: 1 if any(map(str.isdigit, x)) else 0)

        # move sentiment label to last column: THIS SHOULD BE LAST LINE OF CODE BEFORE df.to_csv
        # reference: https://stackoverflow.com/questions/25122099/move-column-by-name-to-front-of-table-in-pandas
        df = df[[col for col in df.columns if col != 'sentiment'] + ['sentiment']]
        df.to_csv(directory + file.split('.')[0] +'_cleaned_features.csv', index=False, encoding='utf-8-sig')
        print('done with', file)

