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
    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret = ""
elif user == "varun":
    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret = ""

t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
# change file name to whatever csv you downloaded from IEEE: https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset
filename = 'corona_tweets_01.csv'
df = pd.read_csv(filename)
truncated = df.head(100)
truncated.to_csv(filename + '_readable.csv')
file = open(filename)
hydrated = t.hydrate(open(filename))
all_users = []
for line in file:
    tweet = t.hydrate(line.split(',')[0])
# for tweet in hydrated:
    try:
        curr_user = dict()
        curr_user['text'] = tweet['text']
        curr_user['id'] = tweet['id']
        if tweet['place']:
            curr_user['place'] = tweet['place']['country']
        else:
            curr_user['place'] = ""
        curr_user['created_at'] = tweet['created_at']
        curr_user['user_location'] = tweet['user']['location']
        curr_user['user_name'] = tweet['user']['name']
        all_users.append(curr_user)
    except:
        continue
print(len(all_users))
df = pd.DataFrame.from_dict(all_users)
print(df.head(10))
outfile = filename.split('.')[0] + '_hydrated.csv'
df.to_csv(outfile)
