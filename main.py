from twarc import Twarc
import pandas as pd
import time
import csv

# this function gets keys and tokens for particular user
# valid inputs: "shan", "hiyori", "varun", "mike", "ori"
def getKeysTokens(user):
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
        consumer_key = "vzyVgjCXw52MNQIP4y3hU9hem"
        consumer_secret = "tFYIw6QIGz8NPiMsyo55cUufb10JyRAHTwnPZsCU9mYgDpb0XH"
        access_token = "1380174793664831493-lxP6ZpqxuaytISermxHYj4HlnIKKwQ"
        access_token_secret = "TucRWP58d296ZOJIyXVHxyzdrvj7tkFU6cbOztN0pnLDN"
    elif user == "ori":
        consumer_key = ""
        consumer_secret = ""
        access_token = ""
        access_token_secret = ""
    elif user == "mike":
        consumer_key = "z8LM5VXrsAhG670TQfojO54Tw"
        consumer_secret = "ltE6fMjMWWlbUrpwvYi0hPOkWNCfBjhmRQkQVRZ9M0Q08rGNzG"
        access_token = "704859952213585920-6BaIraVKCkZJW62qzxBtrWjNQ9YaGkq"
        access_token_secret = "2WwjtoqdDwZ2eyzE8QVAdsCPgCeGH9ORQIC1jG6MphukL"
    else:
        print("invalid username yo, but hit us up if you want to help and make a Twitter dev account lol")
    return [consumer_key, consumer_secret,access_token,access_token_secret]

if __name__ == "__main__":
    # IEEE dataport: https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset
    # retrieve list of csv IDs to split into train, dev, test sets
    num_inputs = input("Type a list of hydrated_sentiment csv file IDs space-delimited, ex: 01 15 100 388: ")
    files = list(map(lambda x: int(x), num_inputs.split()))
    num_files = len(files)
    df_tweets = pd.DataFrame(columns=['text', 'id', 'place', 'created_at', 'user_location', 'user_name',
                                      'followers_count','retweet_count','favorite_count','hashtags'])
    all_users = ["shan","hiyori","mike","varun"]
    # iterate through batch of 3 files inputted
    for file_num in files:
        # extract ids from file
        print('On tweet file number ' + str(file_num) + ' out of ' + str(num_files) + ' total files')
        if file_num < 10:
            filename = 'corona_tweets_0' + str(file_num) + '.csv'
        else:
            filename = 'corona_tweets_' + str(file_num) + '.csv'
        # extract ids to separate csv
        df = pd.read_csv(filename)
        ids=df.iloc[:,0]
        id_file = filename[:-4] + '_ids.csv'
        ids.to_csv(id_file, index=False)
        print("successfully extracted tweet ids, check it out at", id_file)
        all_tweets = []
        df = pd.read_csv(filename, header=None)
        df.columns = ['id', 'sentiment']
        total_tweets = df.shape[0]
        df_tweet_file = pd.DataFrame(
            columns=['text', 'id', 'place', 'created_at', 'user_location', 'user_name', \
                     'followers_count', 'retweet_count', 'favorite_count', 'hashtags'])
        tweet_idx = 0
        user_count = 0 # what index of all_users I am currently using to hydrate
        num_iter = 0 # how many total users I have gone through
        tweet_limit = 300
        target_length = 10000 # get 10,000 tweets from each file
        user = all_users[user_count]
        [consumer_key, consumer_secret,access_token,access_token_secret] = getKeysTokens(user)
        t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
        curr_df = pd.read_csv(id_file)
        all_ids = curr_df.iloc[:,0]
        last_idx = (len(all_ids) - 1)
        print(id_file, "has",last_idx,"ids")
        curr_idx = range(num_iter*tweet_limit,(num_iter+1)*tweet_limit)
        # setup output .csv
        hydrated_file = filename[:-4] + '_hydrated.csv'
        open(hydrated_file,'w')
        print('beginning to hydrate', target_length, 'number of tweets from', id_file)
        # hydrate up to target_length number of tweets
        while num_iter * tweet_limit < target_length + tweet_limit:
            tweets = t.hydrate(all_ids[curr_idx])
            for tweet in tweets:
                try:
                    curr_tweet = dict()
                    # ignore any non-English tweets or tweets that don't contain the word vaccine/vax
                    if tweet['lang'] != "en":
                        continue
                    curr_tweet['text'] = tweet['full_text']
                    curr_tweet['id'] = tweet['id']
                    curr_tweet['place'] = tweet['place']['country'] if tweet['place'] else ""
                    curr_tweet['created_at'] = tweet['created_at']
                    curr_tweet['user_location'] = tweet['user']['location']
                    curr_tweet['user_name'] = tweet['user']['name']
                    curr_tweet['user_followers_count'] = tweet['user']['followers_count']
                    curr_tweet['retweet_count'] = tweet['retweet_count']
                    curr_tweet['favorite_count'] = tweet['favorite_count']
                    curr_tweet['hashtags'] = tweet['entities']['hashtags'][0] if tweet['entities']['hashtags'] else ""
                    all_tweets.append(curr_tweet)
                except Exception as e:
                    print('EXCEPTION:', e)
                    continue
                tweet_idx += 1
            # update number of batches iterated and index of ids to hydrate
            num_iter += 1
            curr_idx = range(num_iter*300,(num_iter+1)*300)
            # cycle to next account if limit is reached
            user_count += 1
            if user_count != len(all_users):
                print("milked", user, "'s limit, onto using ", all_users[user_count], "'s account")
            else:
                print("milked", user, "'s limit, we've now cycled through a total of", num_iter, "users")
                print("take a break! we need to rest 15 minutes before cycling through our accounts again")
                print("you can check out what tweets we've currently hydrated at", hydrated_file)
                time.sleep(60 * 15)
                user_count = 0
            df_tweets = pd.DataFrame.from_dict(all_tweets)
            df_tweets.to_csv(hydrated_file, index=False, mode='a', header=False)
            all_tweets = []
            user = all_users[user_count]
            [consumer_key, consumer_secret, access_token, access_token_secret] = getKeysTokens(user)
            t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
        print("successfully stored hydrated tweets for file ",file_num, "in", hydrated_file)