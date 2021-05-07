import pandas as pd

# this script splits data into 80% train, 10% dev, and 10% test by extracting that ratio from each file and combining
if __name__ == "__main__":
    # retrieve list of csv IDs to split into train, dev, test sets
    num_inputs = input("Type a list of hydrated_sentiment csv file IDs space-delimited, ex: 01 15 100 388: ")
    files = map(lambda x: 'corona_tweets_' + x + '_hydrated_sentiments.csv', num_inputs.split())
    # set up df
    header = ['text', 'id', 'place', 'created_at', 'user_location', 'user_name', 'followers_count', 'retweet_count', 'favorite_count', 'hashtags', 'sentiment']
    # set up export .csv
    train_file = 'train.csv'
    open(train_file, 'w')
    dev_file = 'dev.csv'
    open(dev_file, 'w')
    test_file = 'test.csv'
    open(test_file, 'w')
    train_df = pd.DataFrame(columns=header)
    dev_df = pd.DataFrame(columns=header)
    test_df = pd.DataFrame(columns=header)
    for file in files:
        df = pd.read_csv(file,header=0,low_memory=False)
        num_rows = len(df)
        last_train_row = int(num_rows*0.8)
        train_df = train_df.append(pd.DataFrame(df.loc[1:last_train_row]), ignore_index=True)
        last_dev_row = last_train_row+int(num_rows*0.1)
        dev_df = dev_df.append(pd.DataFrame(df.loc[last_train_row+1:last_dev_row]), ignore_index=True)
        test_df = test_df.append(pd.DataFrame(df.loc[last_dev_row+1:num_rows]), ignore_index=True)
        train_df.to_csv(train_file, index=False, mode='a')
        dev_df.to_csv(dev_file, index=False, mode='a')
        test_df.to_csv(test_file, index=False, mode='a')
        print("Finished splitting data from ", file)
    print("Done! Check out results at train.csv, dev.csv, test.csv")
    print("train.csv number of rows:", len(train_df))
    print("dev.csv number of rows:", len(dev_df))
    print("test.csv number of rows:", len(test_df))
