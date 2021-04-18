import pandas as pd
from better_profanity import profanity

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

        # add other features 
        # presence of numbers 
        df['num_present'] = df['text'].map(lambda x: any(map(str.isdigit, x)))
        # presence of "trump" 
        df['trump_present'] = df['text'].map(lambda x: 'trump' in x.lower())
        # presence of hashtags 
        df['hashtag_present'] = df['hashtags'].map(lambda x: isinstance(x, str))
        # presence of covid/pandemic/quarantine-related words (from IEEE Dataport)
        covid_words = ["corona", "coronavirus", "covid", "covid19", "covid-19", "sarscov2", "sars cov2", "sars cov 2", "covid_19", "ncov", "ncov2019", "2019-ncov", "pandemic", "2019ncov", "quarantine", "lockdown", "social distancing", "strain", "strains", "variant", "variants"]
        df['covid_present'] = df['text'].map(lambda x: any(word in x.lower() for word in covid_words))
        # presence of vaccine-related words 
        vaccine_words = ["vaccine", "vaccines", "corona vaccine", "corona vaccines", "#coronavaccine", "#coronavaccines", "vax", "pfizer", "biontech", "moderna"]
        df['vaccine'] = df['text'].map(lambda x: any(word in x.lower() for word in vaccine_words))
        # text includes profanity or offensive language 
        df['profanity_present'] = df['text'].map(lambda x: profanity.censor(x) != x)

        # move sentiment label to last column: THIS SHOULD BE LAST LINE OF CODE BEFORE df.to_csv
        # reference: https://stackoverflow.com/questions/25122099/move-column-by-name-to-front-of-table-in-pandas
        df = df[[col for col in df.columns if col != 'sentiment'] + ['sentiment']]
        df.to_csv(directory + file.split('.')[0] +'_cleaned_features.csv', index=False, encoding='utf-8-sig')
        print('done with', file)

