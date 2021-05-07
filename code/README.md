# How to run Final Model (GLoVe LSTM)


1.   Place lstm.py in a directory.
2.   Place the data files train_vax_cleaned.csv, dev_vax_cleaned.csv, and test_vax_cleaned.csv in a folder named train-dev-test in the same directory as 1.
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

# How to run unsupervised learning model (LDA)

# How to run unsupervised learning model (sentiment tracking over time)
1. Run Colab from top to bottom: https://colab.research.google.com/drive/11fyVYRmf2BSKmOB3Xx6fToXReCn5wyrG#scrollTo=FbULFBgox0tS
2. You can then visualize data through your favorite platform (Excel, MATLAB, Python matplotlib, etc)


