import csv
from sklearn.metrics import *
from itertools import chain
import pandas as pd

if __name__ == "__main__":
    # get gold labels filename 
    gold_file = input("Type the .csv gold label for test file (ex: gold_test.csv):")

    # read csv
    df = pd.read_csv(gold_file)

    # get majority class and assign it to all tweets
    majority_class = df['sentiment'].value_counts().idxmax()
    df['pred'] = majority_class

    # write to csv
    df.to_csv("_".join(gold_file.split("_")[:3]) + '_pred.csv', index=False, columns=['pred'])
    print("Check out the predictions in " + "_".join(gold_file.split("_")[:3]) + "_pred.csv")
    df.to_csv(gold_file.split(".")[0] + '_pred.csv', index=False)
    print("Check out the IDs, gold labels, and predictions in " + gold_file.split(".")[0] + "_pred.csv")