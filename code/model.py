# This is the model file where we run the implemented log reg model
# Wall of imports
# Import statements 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import string
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import f1_score

# Load data
def load_data(train_file, dev_file, test_file):
    # Load from CSV
    train_feats = pd.read_csv(train_file)
    dev_feats = pd.read_csv(dev_file)
    test_feats = pd.read_csv(test_file)

    # Clear whitespace just in case 
    train_feats.columns = train_feats.columns.str.strip()
    dev_feats.columns = dev_feats.columns.str.strip()
    test_feats.columns = test_feats.columns.str.strip()

    return train_feats, dev_feats, test_feats

# Clean data
def clean_data(df):
    # Drop all rows where time is NA?
    df = df[df['time'].notna()]

    # Just drop hashtags for now, too difficult for M3 to deal with
    df = df.drop(['hashtags'], axis=1)

    # Change sentiment labels to numbers!
    df['sentiment'] = df['sentiment'].astype('category')
    df["sentiment"] = df["sentiment"].cat.codes

    # Change location
    df["user_location"] = df["user_location"].astype('category')
    df["user_location"] = df["user_location"].cat.codes

    # Change place
    df["place"] = df["place"].astype('category')
    df["place"] = df["place"].cat.codes

    # Change weekday
    #df["weekday"] = df["weekday"].astype('category')
    #df["weekday"] = df["weekday"].cat.codes

    # Change month
    df["month"] = df["month"].astype('category')
    df["month"] = df["month"].cat.codes

    # Change time to datetime object
    #df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(str(x), '%H:%M:%S'))

    # Add hour and minute columns
    #df['hour'] = df['time'].apply(lambda x: x.hour)
    #df['minute'] = df['time'].apply(lambda x: x.minute)

    # Drop time
    df = df.drop(['time'], axis=1)

    # Change all the bool types to numeric
    df["num_present"] = df["num_present"].astype(int)
    df["trump_present"] = df["trump_present"].astype(int)
    #df["hashtag_present"] = df["hashtag_present"].astype(int)
    df["covid"] = df["covid"].astype(int)
    df["vaccine"] = df["vaccine"].astype(int)
    df["profanity_present"] = df["profanity_present"].astype(int)
    #df["emoji_present"] = df["emoji_present"].astype(int)
    df["url_present"] = df["url_present"].astype(int)
    #df["question_exclamation_present"] = df["question_exclamation_present"].astype(int)

    # Get text features! TF-IDF
    all_text = df['text'].tolist()
    vectorizer = TfidfVectorizer(max_features=100, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(all_text).toarray()

    return processed_features

# Returns properly formatted data (in numpy) for sklearn
# This may seem trivial, but we need this when we use more non-text features
def convert_data(df, processed_features):
    #non_text_feats = df.to_numpy()
    #X = np.concatenate((processed_features, non_text_feats), axis=1)
    # Get label
    labels = df['sentiment']

    X = processed_features
    y = labels.to_numpy()

    return X, y

def train_model(X, y):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)

    return model

def model_stats(model, X_train, y_train, X_dev, y_dev, X_test, y_test):
    y_preds_train = model.predict(X_train)
    y_preds_dev = model.predict(X_dev)
    y_preds_test = model.predict(X_test)

    print("Training F1: ", f1_score(y_train, y_preds_train, average='micro'))
    print("Dev F1: ", f1_score(y_dev, y_preds_dev, average='micro'))
    print("Test F1: ", f1_score(y_test, y_preds_test, average='micro'))

# Run everything!
if __name__ == '__main__':
    # Define raw data here
    train_file = 'train-dev-test/train_cleaned_features.csv'
    dev_file = 'train-dev-test/dev_cleaned_features.csv'
    test_file = 'train-dev-test/test_cleaned_features.csv'

    # Load data
    print("Loading Data")
    train_feats, dev_feats, test_feats = load_data(train_file, dev_file, test_file)

    # Temp step - drop the NAs where there is no time
    train_feats = train_feats[train_feats['time'].notna()]
    dev_feats = dev_feats[dev_feats['time'].notna()]
    test_feats = test_feats[test_feats['time'].notna()]

    # Clean data
    print("Cleaning Data")
    train_feats_processed = clean_data(train_feats)
    dev_feats_processed = clean_data(dev_feats)
    test_feats_processed = clean_data(test_feats)

    # Convert data into proper format
    print("Converting Data")
    X_train, y_train = convert_data(train_feats, train_feats_processed)
    X_dev, y_dev = convert_data(dev_feats, dev_feats_processed)
    X_test, y_test = convert_data(test_feats, test_feats_processed)

    # Train model
    print("Training Model")
    model = train_model(X_train, y_train)

    # Output stats
    model_stats(model, X_train, y_train, X_dev, y_dev, X_test, y_test)