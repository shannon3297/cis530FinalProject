import pandas as pd
import numpy as np

if __name__ == "__main__":
    # get file IDs
    input_nums = input("Type a list of hydrated_sentiment/IEEE downloaded csv file IDs space-delimited, ex: 01 15 100 388:")
    file_nums = map(int, input_nums.split())
    # join df for all files inputted
    for file_num in file_nums:
        if file_num < 10:
            hydrated_file = 'corona_tweets_0' + str(file_num)+ '_hydrated.csv'
            sentiment_id_file = 'corona_tweets_0' + str(file_num) + '.csv'
        else:
            hydrated_file = 'corona_tweets_' + str(file_num) + '_hydrated.csv'
            sentiment_id_file = 'corona_tweets_' + str(file_num) + '.csv'

        # read files
        hydrated_df = pd.read_csv(hydrated_file, header=None)
        sentiment_id_df = pd.read_csv(sentiment_id_file, header=None)

        # set column names
        hydrated_df.columns = ['text', 'id', 'place', 'created_at', 'user_location', 'user_name', 'followers_count', 'retweet_count', 'favorite_count', 'hashtags']
        sentiment_id_df.columns = ['id', 'sentiment']

        # change sentiments from numbers to neg: [-1,0), neutral: 0, positive: (0,1]
        sentiment_id_df['sentiment'] = pd.to_numeric(sentiment_id_df['sentiment'], downcast='float')
        criteria = [sentiment_id_df['sentiment'] < 0, sentiment_id_df['sentiment'] == 0, sentiment_id_df['sentiment'] > 0]
        values = ['negative', 'neutral', 'positive']
        sentiment_id_df['sentiment'] = np.select(criteria, values, 'NA')

        # join hydrated_df with sentiment_id_df
        joined_df = hydrated_df.join(sentiment_id_df.set_index('id'), on='id', how='left')

        # write to CSV
        joined_file = hydrated_file.split(".")[0] + '_sentiments.csv'
        joined_df.to_csv(joined_file, index=False)
        print('Check out joined file at ', joined_file)
    # folder got too crazy with so many outputs
    # joined_df['sentiment'].to_csv(sentiment_id_file.split(".")[0] + '_sentiments.csv', index=False)
    # print('Check out gold label sentiments at ' + sentiment_id_file.split(".")[0] + '_sentiments.csv')
    # joined_df.to_csv(sentiment_id_file.split(".")[0] + '_id_sentiments.csv', index=False, columns=['id', 'sentiment'])
    # print('Check out IDs and gold label sentiments at ' + sentiment_id_file.split(".")[0] + '_id_sentiments.csv')
