# importing the Python module
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# print(label_names)
# print(labels)
# print(feature_names)


# splitting the data
train, test, train_labels, test_labels = train_test_split(features, labels,
                                       test_size = 0.20, random_state = 42)

gaus_cf = GaussianNB()
model = gaus_cf.fit(train, train_labels)

# making the predictions
y_pred = gaus_cf.predict(test)

# print(y_pred)

print(accuracy_score(test_labels, y_pred))
