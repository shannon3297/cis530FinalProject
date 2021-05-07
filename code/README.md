# How to Get Data
1. Download our data from here (only use the three files): https://drive.google.com/drive/folders/1VIzIzGbztzgnkBygpBsKWGlUxYGzRBlY?usp=sharing :). This folder, which should have be called 'data' and be placed in the same folder as all the other code, will have only 3 files (train.csv, dev.csv, test.csv). 

OR (if you're up for a challenge)

2. Hydrate Tweets (extract tweets from scratch)

The term "hydrate" is used to describe the process of extracting content from Tweets and various metadata with a Twitter dev account given list of Tweet IDs. We consulted this guide https://theneuralblog.com/hydrating-tweet-ids/ and included our code snippet in the Appendix section of the final report. To extract the data, download csv of tweet IDs from target date from IEEE Dataport(https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset) into the same folder that extract-data.py script lives in and run the extract-data.py script. It will ask you to input the ID of the downloaded csv and begin hydrating Tweet IDs located in that csv using 4 of our Twitter dev accounts. If you want to speed up runtime, you can add your Twitter user info in getKeysTokens() function. We are limited to hydrating 300 tweets per user in 15 minutes which is why extracting the data was no quick feat! 

We used various scripts to extract and clean our data: cleaning-features.py, extract_data.py, sentiment-join-extract.py, train_dev_test.py


# How to run Final Model (GLoVe LSTM) - Extension 1

NOTE: there appears to be some issues with running this code locally on Windows as it was developed on a Linux machine. If you have trouble running the code when following the instructions below, please upload the lstm.ipynb file onto Google Colab and run all cells. You will need to mount your Google Drive (by running the "mount google drive" cell) and place the data and glove.6B.50d.txt files in /content/drive/MyDrive/CIS530-project/. Otherwise if running locally, please use the following instructions:

1.   Place lstm.py in the code/ directory.
2.   Place the data files train.csv, dev.csv, and test.csv in a folder named data in the same root directory as 1.
3.   Download glove.6B.50d.txt file from https://nlp.stanford.edu/projects/glove/ and place it in the same directory as 1.  
4.   Run the following commands in the command line to download the necessary packages and models:


```
pip install swifter
pip install torch
pip install spacy
pip install seaborn
python -m spacy download en 
```

5. Run the following command to run the model to reproduce the report results:

```
python lstm.py
```

# How to run unsupervised learning model (LDA) - Extension 2

NOTE: This file caused issues on some machines, mostly because of the old packages/libraries it uses. If an error occurs with the .py file, please run the notebook ("lda_sentiment.ipynb") instead!

1. Download lda_sentiment.py/ipynb
2. Download data into a subfolder within the same folder that lda_sentiment.py is in 
3. Confirm the data folder has train.csv, dev.csv, test.csv
4. Run lda_sentiment.py (input path to train/dev/test, should be data/train.csv, data/dev.csv, and data/test.csv if you set up structure correctly)
5. You can then visualize the clusters!

# How to get some fun insights (sentiment tracking over time) - Extension 2
1. Download trending_sentiment.py
2. Download data into a subfolder within the same folder that trending_sentiment.py is in
3. Confirm the data folder has train.csv, dev.csv, test.csv
4. Run trending_sentiment.py (input path to train/dev/test, should be data/train.csv, dev/train.csv, and test/train.csv if you set up structure correctly)
5. You can then visualize data through your favorite platform (Excel, MATLAB, Python matplotlib, etc)

# How to run evaluation script
1.    From the directory that includes score.py, run "python score.py". 
2.    For each prompt that shows up, input the correct paths of the two files (gold label file and predicted labels file)
3.    The output for the provided test gold and predicted labels should look like this:

*    negative: 0.45454545
*    neutral: 0.80066348
*    positive: 0.77755529
*    weighted average: 0.734895155
