# Wall of imports (mostly bc of viz)
import pandas as pd
import string
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import f1_score
import gensim.corpora as corpora
import re
from pprint import pprint
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud

# Load data
train_feats = pd.read_csv('data/train.csv', delimiter=',', encoding="utf-8")
dev_feats = pd.read_csv('data/dev.csv', delimiter=',', encoding="utf-8")
test_feats = pd.read_csv('data/test.csv', delimiter=',', encoding="utf-8")

# Clear whitespace
train_feats.columns = train_feats.columns.str.strip()

# Dropping stuff!

# Drop all rows where time is NA?
print("Old length: ", len(train_feats))
train_feats = train_feats[train_feats['time'].notna()]
print("New length: ", len(train_feats))

# Just drop hashtags for now, too difficult for M3 to deal with
train_feats = train_feats.drop(['hashtags'], axis=1)

# Feature manipulation to make them ready for the model
# Change sentiment labels to numbers!
train_feats['sentiment'] = train_feats['sentiment'].astype('category')
train_feats["sentiment"] = train_feats["sentiment"].cat.codes

# Change location
train_feats["user_location"] = train_feats["user_location"].astype('category')
train_feats["user_location"] = train_feats["user_location"].cat.codes

# Change place
train_feats["place"] = train_feats["place"].astype('category')
train_feats["place"] = train_feats["place"].cat.codes

# Change weekday
#train_feats["weekday"] = train_feats["weekday"].astype('category')
#train_feats["weekday"] = train_feats["weekday"].cat.codes

# Change month
train_feats["month"] = train_feats["month"].astype('category')
train_feats["month"] = train_feats["month"].cat.codes

# Change time to datetime object
#train_feats['time'] = train_feats['time'].apply(lambda x: datetime.datetime.strptime(str(x), '%H:%M:%S'))

# Add hour and minute columns
#train_feats['hour'] = train_feats['time'].apply(lambda x: x.hour)
#train_feats['minute'] = train_feats['time'].apply(lambda x: x.minute)

# Drop time
train_feats = train_feats.drop(['time'], axis=1)

# Change all the bool types to numeric
train_feats["num_present"] = train_feats["num_present"].astype(int)
train_feats["trump_present"] = train_feats["trump_present"].astype(int)
#train_feats["hashtag_present"] = train_feats["hashtag_present"].astype(int)
train_feats["covid"] = train_feats["covid"].astype(int)
train_feats["vaccine"] = train_feats["vaccine"].astype(int)
train_feats["profanity_present"] = train_feats["profanity_present"].astype(int)
#train_feats["emoji_present"] = train_feats["emoji_present"].astype(int)
train_feats["url_present"] = train_feats["url_present"].astype(int)
#train_feats["question_exclamation_present"] = train_feats["question_exclamation_present"].astype(int)

# Remove punctuation
train_feats['text'] = \
train_feats['text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
train_feats['text'] = \
train_feats['text'].map(lambda x: x.lower())
# Print out the first rows of papers
train_feats['text'].head()

# More processing of data
stop_words = stopwords.words('english')
# Add these common stop words from Tweets
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'https', 'tco', 'rt', 'don\'', 'rsgt', 't', 'vaccine',
'vaccines', 'amp', 'covid'])

# Two helper functions that helps with transforming the words
def sentence_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

# Extract text data into a list
data = train_feats.text.values.tolist()
data_words = list(sentence_to_words(data))

# Remove stop words finally
data_words = remove_stopwords(data_words)

# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# number of topics
num_topics = 6
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 6 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

# Output the final word clouds
cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()