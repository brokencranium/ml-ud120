#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
y, X = targetFeatureSplit(data)


###############################################################################
# Split into a training and testing set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

###############################################################################
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import RandomizedPCA

n_components = 1

# k_fold = KFold(n_splits=10)
#
# for X_train, X_test in k_fold.split(features, labels):
#     pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
#     X_train_pca = pca.transform()
#     X_test_pca = pca.transform(X_test)

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                                                     test_size=0.3, random_state=42)
# pca = PCA(n_components=n_components)
# pca.fit(X_train)
# X_t_train = pca.transform(X_train)
# X_t_test = pca.transform(X_test)

# clf = SVC()
# clf.fit(X_t_train, y_train)
# print 'score', clf.score(X_t_test, y_test)
# print 'pred label', clf.predict(X_t_test)


clf_dtc = DecisionTreeClassifier(random_state=42)
clf_dtc.fit(X_train, y_train)
y_pred = clf_dtc.predict(X_test)
print 'DTC score', clf_dtc.score(X_test, y_test)
print 'DTC pred label', clf_dtc.predict(X_test)

count = 0
for t in y_test:
    if t > 0:
        count = count + 1
print count


# true_positive = positive / len(y_test)


# precision
from sklearn.metrics import precision_score
print precision_score(y_pred, y_test)

#recall
from sklearn.metrics import recall_score
print recall_score(y_test, y_pred)


# true positive
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for i in range(0, len(predictions)):
    if true_labels[i] == predictions[i] and true_labels[i] == 1:
        true_positive = true_positive + 1
    if true_labels[i] == predictions[i] and true_labels[i] == 0:
        true_negative = true_negative + 1
    if true_labels[i] == 0 and predictions[i] == 1:
        false_positive = false_positive + 1
    if true_labels[i] == 1 and predictions[i] == 0:
        false_negative = false_negative + 1

print("True positive", true_positive)
print("True negative", true_negative)
print("False positive", false_positive)
print("False negative", false_negative)
print precision_score(true_labels, predictions)
print recall_score(true_labels, predictions)

from sklearn.metrics import f1_score
print f1_score(true_labels,predictions)
# it's all yours from here forward!
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svr = svm.SVC()
# clf = GridSearchCV(svr, parameters)
# clf.fit(iris.data, iris.target)
