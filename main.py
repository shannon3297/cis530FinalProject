from twarc import Twarc
import pandas as pd
from urllib.error import HTTPError
from datetime import datetime 
import time 

user = "shan" # change to varun or hiyori and update your keys/tokens
if user == "shan":
    consumer_key="GSXDGhLhvzCoC2Uh5rxdAYEQr"
    consumer_secret="heoSJn2ApCIGeSLVXfweRMvX3YPAnVvMYThvfdke9uqTvpMjBn"
    access_token="3622661658-PdlAnhD3Jkh7Tgs27mygSJ0EItIvfvniR977Y4S"
    access_token_secret="y1MWurpwIPVVlcCqRd169yWcob8B1KKvg2vnl49dyQncV"
elif user == "hiyori":
    consumer_key = "gCBvdmiVMt0gkuxdKeRzGMgWR"
    consumer_secret = "qUpB8Gmp3zKwfjukD6HYYuVSAdnXnXKAsMqAF1ayIja9uB2Rnx"
    access_token = "1380171300073832448-CFDU1goMdSK4eCisjDEgKG6pAkuJYl"
    access_token_secret = "EHI7kRnHcbFy507q84jOCJ7kDf8uqMkfjFd59xK7jdrU3"
elif user == "varun":
    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret = ""

t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
# change file name to whatever csv you downloaded from IEEE: https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset
first_file_num = 1 # TODO Change to the first file you're doing 
last_file_num = 1 # TODO Change to last file 
file_range = list(range(first_file_num, last_file_num + 1))
num_files = len(file_range)
df_tweets = pd.DataFrame(columns = ['text', 'id' ,'place', 'created_at', 'user_location', 'user_name', 'hashtags', 'sentiment'])
for num in file_range:
    print('On file ' + str(num) + ' out of ' + str(num_files))
    all_tweets = []
    if num < 10:
        filename = 'corona_tweets_0' + str(num) + '.csv'
    else:
        filename = 'corona_tweets_' + str(num) +'.csv'
    df = pd.read_csv(filename, header=None)
    df.columns = ['id', 'sentiment']
    total_tweets = df.shape[0]
    # df.sort_values(df.columns[0], inplace=True)
    last_tweet = 0 
    df_tweet_file = pd.DataFrame(columns = ['text', 'id' ,'place', 'created_at', 'user_location', 'user_name', 'hashtags'])
    while last_tweet < total_tweets - 1:
        batch_size = min(5, total_tweets - last_tweet) # TODO change 5 to 300 
        ids = df['id'][last_tweet:last_tweet+batch_size]
        print(ids)
        ids.to_csv(filename.split('.')[0] + '_ids.csv', index=False, header=False)
        time.sleep(10)
        # file = open(filename)
        # hydrated = t.hydrate(open(filename))
        start_time = datetime.now()
        df_2 = pd.read_csv(filename.split('.')[0] + '_ids.csv', header=None)
        print(df_2)
        for tweet in t.hydrate(open(filename.split('.')[0] + '_ids.csv')):
            print(tweet)
            # tweet = t.hydrate(line.split(',')[0])
        # for tweet in hydrated:
            try:
                print('in try')
                curr_tweet = dict()
                # print(tweet['entities'])
                curr_tweet['text'] = tweet['full_text']
                # print(tweet['full_text'])
                curr_tweet['id'] = tweet['id']
                if tweet['place']:
                    curr_tweet['place'] = tweet['place']['country']
                else:
                    curr_tweet['place'] = ""
                curr_tweet['created_at'] = tweet['created_at']
                curr_tweet['user_location'] = tweet['user']['location']
                curr_tweet['user_name'] = tweet['user']['name']
                if tweet['entities']['hashtags']:
                    curr_tweet['hashtags'] = tweet['entities']['hashtags'][0]
                else:
                    curr_tweet['hashtags'] = ""
                all_tweets.append(curr_tweet)
                print('ALL', all_tweets)
            except Exception as e:
                print('EXCEPTION:', e)
                continue
        df_tweets_add = pd.DataFrame.from_dict(all_tweets)
        print('DF_TWEETS_ADD', df_tweets_add)
        # df_tweets_add.sort_values(['id'], inplace=True)
        # print(df_tweets_add)
        # df_tweets_add['sentiment'] = df.head(batch_size).iloc[:,1]
        # print(df_tweets_add)
        df_tweet_file = df_tweet_file.append(df_tweets_add)
        print(df_tweet_file)
        last_tweet += batch_size
        print('LAST', last_tweet)
        if num != last_file_num and last_tweet != total_tweets:
            end_time = datetime.now()
            diff = end_time - start_time
            time.sleep(10 - diff.total_seconds())
    df_tweets_add = df_tweet_file.join(df.set_index('id'), on='id', lsuffix='l', rsuffix='r')
    print(df_tweets_add)
    df_tweets.append(df_tweets_add)
# print(len(all_tweets))
# print(df_tweets.head(10))
outfile = 'corona_tweets_' + str(first_file_num) + '-' + str(last_file_num) + '_hydrated.csv'
df_tweets.to_csv(outfile, index=False)
