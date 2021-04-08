from twarc import Twarc
import pandas as pd
from urllib.error import HTTPError

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
filename = 'corona_tweets_01.csv'
df = pd.read_csv(filename)
batch_size = 50
ids = df.head(batch_size).iloc[:,0]
ids.to_csv(filename[:-4] + '_ids.csv', index=False)
# file = open(filename)
# hydrated = t.hydrate(open(filename))
all_tweets = []
for tweet in t.hydrate(open(filename[:-4] + '_ids.csv')):
    # tweet = t.hydrate(line.split(',')[0])
# for tweet in hydrated:
    try:
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
    except Exception as e:
        print('EXCEPTION:', e)
        continue
# print(len(all_tweets))
df_tweets = pd.DataFrame.from_dict(all_tweets)
df_tweets['sentiment'] = df.head(batch_size).iloc[:,1]
print(df_tweets.head(10))
outfile = filename.split('.')[0] + '_hydrated.csv'
df_tweets.to_csv(outfile, index=False)
