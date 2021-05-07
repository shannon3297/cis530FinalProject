import csv
from sklearn.metrics import *
from itertools import chain
import pandas as pd

if __name__ == "__main__":
    # get gold labels and test file filename 
    gold_file = input("Type the .csv gold label for training file (ex: gold_train.csv):")
    test_file = input("Type the filename of the test file (ex: test.csv):")

    # read csv
    df = pd.read_csv(gold_file)
    df_test = pd.read_csv(test_file)

    # get majority class and assign it to all tweets in train and test 
    counts = df['sentiment'].value_counts()
    print(counts)
    majority_class = counts.idxmax()
    df['pred'] = majority_class
    df_test['pred'] = majority_class

    # write to csv
    df.to_csv("_".join(gold_file.split("_")[:3]) + '_pred_train.csv', index=False, columns=['pred'])
    print("Check out the TRAIN SET predictions in " + "_".join(gold_file.split("_")[:3]) + "_pred_train.csv")
    df.to_csv("_".join(gold_file.split("_")[:3]) + '_pred_train_full.csv', index=False)
    print("Check out the TRAIN SET IDs, gold labels, and predictions in " + "_".join(gold_file.split("_")[:3]) + "_pred_train_full.csv")

    df_test.to_csv("_".join(test_file.split("_")[:3]) + '_pred_test.csv', index=False, columns=['pred'])
    print("Check out the TEST SET predictions in " + "_".join(test_file.split("_")[:3]) + "_pred_test.csv")
    df_test.to_csv("_".join(test_file.split("_")[:3]) + '_pred_test_full.csv', index=False)
    print("Check out the TEST SET IDs, gold labels, and predictions in " + "_".join(test_file.split("_")[:3]) + "_pred_test_full.csv")