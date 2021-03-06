#!/usr/bin/python

from __future__ import division

import pickle
import sys

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
# from sklearn.model_selection import GridSearchCV

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus',
#                  'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi',
#                  'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
#                  'from_messages', 'other', 'from_this_person_to_poi', 'director_fees',
#                  'deferred_income', 'long_term_incentive',
#                  'from_poi_to_this_person']

features_list = ['poi',
                 'salary',
                 'total_payments',
                 'bonus',
                 'restricted_stock',
                 'total_stock_value',
                 'expenses',
                 'other',
                 'long_term_incentive',
                 'shared_receipt_with_poi',
                 'from_poi_ratio',
                 'to_poi_ratio'
                 ]

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
data_dict.pop('TOTAL')

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = {}
poi_dict = {}
scale_factor = 1

for key, val in data_dict.iteritems():
    if val['total_payments'] > 100000000 and val['total_payments'] != 'NaN':
        continue

    if val['poi']:
        poi_dict[key] = val

    try:
        if int(val['from_messages']) < int(val['from_poi_to_this_person']):
            val['from_messages'] = val['from_poi_to_this_person']

        if int(val['to_messages']) < int(val['from_poi_to_this_person']):
            val['to_messages'] = val['from_this_person_to_poi']

        val['from_poi_ratio'] = scale_factor * int(val['from_poi_to_this_person']) / int(
            val['from_messages'])
        val['to_poi_ratio'] = scale_factor * int(val['from_this_person_to_poi']) / int(
            val['to_messages'])
    except (ZeroDivisionError, ValueError):
        if val['poi']:
            val['from_poi_ratio'] = 1 * scale_factor
            val['to_poi_ratio'] = 1 * scale_factor
        if not val['poi']:
            val['from_poi_ratio'] = 0
            val['to_poi_ratio'] = 0

    my_dataset[key] = val

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1)

# scaler = preprocessing.MinMaxScaler()
# features_train = scaler.fit_transform(features_train)

clf_s = SVC(kernel='sigmoid')
clf_s.fit(features_train, labels_train)
predict_r = clf_s.predict(features_test)
print("SVC")
print(classification_report(labels_test, predict_r))

clf_r = RandomForestClassifier(max_depth=7, random_state=0, oob_score=True)
clf_r.fit(features_train, labels_train)
predict_r = clf_r.predict(features_test)
print("Random Forest")
print(classification_report(labels_test, predict_r))

clf_d = tree.DecisionTreeClassifier()
clf_d.fit(features_train, labels_train)
predict_d = clf_d.predict(features_test)
print("Decision Tree Classifier")
print(classification_report(labels_test, predict_d))
print(cross_val_score(estimator=clf_d, X=features_train, y=labels_train, cv=7, n_jobs=4))

# parameters = {'max_depth':range(3, 20)}
# clf_d = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
# clf_d.fit(X=features_train, y=labels_train)
# clf_d_est = clf_d.best_estimator_
# print (clf_d.best_score_, clf_d.best_params_)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_d, my_dataset, features_list)
