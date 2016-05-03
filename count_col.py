import pandas as pd
from sklearn import metrics

training = pd.read_csv(open("training_set.csv"))
#testing = pd.read_csv(open("testing_set.csv"))



print len(training.columns)
