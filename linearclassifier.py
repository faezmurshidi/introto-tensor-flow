from tensorflow.contrib import skflow
import pandas as pd
from sklearn import metrics

training = pd.read_csv(open("training_set.csv"))
testing = pd.read_csv(open("testing_set.csv"))

x =len(training.columns)

training_data = training.ix[:,0:x-2]
training_target = training.ix[:,x-1]
testing_data = testing.ix[:,0:x-2]


classifier = skflow.TensorFlowLinearClassifier(n_classes=x-2)


classifier.fit(training_data, training_target)

predicted = classifier.predict(testing_data)

id = pd.DataFrame(list(range(1, len(predicted) + 1)),columns=['Id'])
predicted = pd.DataFrame(predicted,columns=['Prediction'])
output = pd.concat([id, predicted], axis=1)

pd.DataFrame(output).to_csv("predictionLC.csv",index=False)


accuracy = classifier.score(training_data, predicted[:13072])
print accuracy

print 'done'
