# How to Get Data
1. Hydrate Tweets (extract tweets from scratch)
The term "hydrate" is used to describe the process of extracting content from Tweets and various metadata with a Twitter dev account given list of Tweet IDs. We consulted this guide https://theneuralblog.com/hydrating-tweet-ids/ and included our code snippet in the Appendix section of the final report. To extract the data, download csv of tweet IDs from target date from IEEE Dataport(https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset) into the same folder that extract-data.py script lives in and run the extract-data.py script. It will ask you to input the ID of the downloaded csv and begin hydrating Tweet IDs located in that csv using 4 of our Twitter dev accounts. If you want to speed up runtime, you can add your Twitter user info in getKeysTokens() function. We are limited to hydrating 300 tweets per user in 15 minutes which is why extracting the data was no quick feat! 

OR 

2. Download our data from here: https://drive.google.com/drive/folders/1VIzIzGbztzgnkBygpBsKWGlUxYGzRBlY?usp=sharing :)

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
1. Download lda_sentiment.py
2. Download data into a subfolder within the same folder that lda_sentiment.py is in 
3. Confirm the data folder has train.csv, dev.csv, test.csv
4. Run lda_sentiment.py (input path to train/dev/test, should be data/train.csv, data/dev.csv, and data/test.csv if you set up structure correctly)
5. You can then visualize the clusters!

# How to run unsupervised learning model (sentiment tracking over time) - Extension 2
1. Download trending_sentiment.py
2. Download data into a subfolder within the same folder that trending_sentiment.py is in
3. Confirm the data folder has train.csv, dev.csv, test.csv
4. Run trending_sentiment.py (input path to train/dev/test, should be data/train.csv, dev/train.csv, and test/train.csv if you set up structure correctly)
5. You can then visualize data through your favorite platform (Excel, MATLAB, Python matplotlib, etc)


