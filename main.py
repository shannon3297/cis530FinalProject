from twarc import Twarc
import pandas as pd
import time

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
        consumer_key = ""
        consumer_secret = ""
        access_token = ""
        access_token_secret = ""
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
    first = input("Type the first file to extract data from (ex: corona_tweets_01.csv = 01):")
    first_file_num = int(first)
    last = input("Type the last file to extract data from (ex: corona_tweets_12.csv = 12):")
    last_file_num = int(last)
    file_range = list(range(first_file_num, last_file_num + 1))
    num_files = len(file_range)
    df_tweets = pd.DataFrame(columns=['text', 'id', 'place', 'created_at', 'user_location', 'user_name',
                                      'followers_count','retweet_count','favorite_count','hashtags', 'sentiment'])
    all_users = ["shan","hiyori","mike"] # TODO: add ,"ori","varun" once they add their info in getKeysTokens
    # iterate through all files [first, last] range
    for num in file_range:
        # extract ids from file
        print('On tweet file number ' + str(num) + ' out of ' + str(num_files) + ' total files')
        if num<10:
            filename = 'corona_tweets_0' + str(num) + '.csv'
        else:
            filename = 'corona_tweets_' + str(num) + '.csv'
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
        num_tweets_hydrated = 0
        user_count = 0 # what index of all_users I am currently using to hydrate
        num_iter = 0 # how many total users I have gone through
        tweet_limit = 300
        user = all_users[user_count]
        [consumer_key, consumer_secret,access_token,access_token_secret] = getKeysTokens(user)
        t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
        curr_df = pd.read_csv(id_file)
        all_ids = curr_df.iloc[:,0]
        last_idx = (len(all_ids) - 1)
        print(id_file, "has",last_idx,"ids")
        curr_idx = range(num_iter*300,(num_iter+1)*300)
        out_file = filename[:-4] + '_hydrated.csv'
        open(out_file,'w')
        # write sentiment (label) to separate file
        df.iloc[:, 1].to_csv(filename[:-4] + '_sentiment.csv', index=False, header=False)
        # hydrate all ids in id.csv
        while last_idx not in curr_idx:
            tweets = t.hydrate(all_ids[curr_idx])
            for tweet in tweets:
                try:
                    curr_tweet = dict()
                    # ignore any non-English tweets or retweets
                    if tweet['lang'] != "en":
                        continue
                    curr_tweet['text'] = tweet['full_text']
                    curr_tweet['id'] = tweet['id']
                    if tweet['place']:
                        curr_tweet['place'] = tweet['place']['country']
                    else:
                        curr_tweet['place'] = ""
                    curr_tweet['created_at'] = tweet['created_at']
                    curr_tweet['user_location'] = tweet['user']['location']
                    curr_tweet['user_name'] = tweet['user']['name']
                    curr_tweet['user_followers_count'] = tweet['user']['followers_count']
                    curr_tweet['retweet_count'] = tweet['retweet_count']
                    curr_tweet['favorite_count'] = tweet['favorite_count']
                    if tweet['entities']['hashtags']:
                        curr_tweet['hashtags'] = tweet['entities']['hashtags'][0]
                    else:
                        curr_tweet['hashtags'] = ""
                    all_tweets.append(curr_tweet)
                    num_tweets_hydrated += 1
                except Exception as e:
                    print('EXCEPTION:', e)
                    continue
                num_iter += 1
                curr_idx = range(num_iter*300,(num_iter+1)*300)
                # we are limited to 300 tweets every 15 min for every account, cycle to next account if limit is reached
                if num_tweets_hydrated == tweet_limit:
                    user_count += 1
                    if user_count != len(all_users):
                        print("milked", user, "'s limit, onto using ", all_users[user_count], "'s account")
                    else:
                        print("take a break! we need to rest 15 minutes before cycling through our accounts again")
                        df_tweets = pd.DataFrame.from_dict(all_tweets)
                        df_tweets.to_csv(out_file, index=False, mode='a', header=False)
                        all_tweets = []
                        print("you can check out what tweets we've currently hydrated at", out_file)
                        exit # uncommment this out to run the cycle for real
                        time.sleep(60 * 15)
                        user_count = 0
                    num_tweets_hydrated = 0
                    user = all_users[user_count]
                    [consumer_key, consumer_secret, access_token, access_token_secret] = getKeysTokens(user)
                    t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
        print("successfully extracted tweets and attributes, check it out at", out_file)