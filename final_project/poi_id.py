#!/usr/bin/python

from __future__ import division

import pickle
import sys

from sklearn import preprocessing

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'bonus',
#                  'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi',
#                  'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances',
#                  'from_messages', 'other', 'from_this_person_to_poi', 'director_fees',
#                  'deferred_income', 'long_term_incentive',
#                  'from_poi_to_this_person']

features_list = ['poi', 'salary', 'total_payments', 'exercised_stock_options', 'bonus',
                 'restricted_stock', 'total_stock_value', 'expenses', 'other', 'deferred_income',
                 'long_term_incentive', 'shared_receipt_with_poi',
                 'from_messages', 'to_messages', 'from_this_person_to_poi',
                 'from_poi_to_this_person',
                 'from_poi_ratio', 'to_poi_ratio']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
data_dict.pop('TOTAL')

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = {}
poi_dict = {}

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

        val['from_poi_ratio'] = int(val['from_poi_to_this_person']) / int(val['from_messages'])
        val['to_poi_ratio'] = int(val['from_this_person_to_poi']) / int(val['to_messages'])
    except (ZeroDivisionError, ValueError):
        if val['poi']:
            val['from_poi_ratio'] = 1
            val['to_poi_ratio'] = 1
        if not val['poi']:
            val['from_poi_ratio'] = 0
            val['to_poi_ratio'] = 0

    my_dataset[key] = val

# Extract features and labels from dataset for local testing
poi_data = featureFormat(poi_dict, features_list, sort_keys=True)
poi_labels, poi_features = targetFeatureSplit(poi_data)

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# clf = GaussianNB()

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3)

features_train.extend(poi_features)
labels_train.extend(poi_labels)

scaler = preprocessing.MinMaxScaler()
features_train = scaler.fit_transform(features_train)

# Need to get this working
# features_train = preprocessing.MaxAbsScaler(features_train).fit(features_train, labels_train)
# features_test = preprocessing.MaxAbsScaler(features_train)

# Takes forever
# normalizer = preprocessing.Normalizer().fit(features_train)
# features_train = normalizer.transform(features_train)

# pca.fit(features_train)
# n_components = [2, 6, 12]
# kernel = [['rbf']]
# class_weight = [['balanced']]
pca = PCA()
clf = SVC()
estimators = [('reduce_dim', pca), ('clf', clf)]
scores = ['precision', 'recall']
pipe = Pipeline(estimators)
# pipe.set_params(clf__C=[0.1, 10, 100])
# pipe.set_params(reduce_dim__n_components=[2, 5, 10])
# param_grid = dict(reduce_dim__n_components=[2, 5, 10],
#                   clf__C=[0.1, 10, 100])
# clf__C = [1e3, 5e3, 1e4, 5e4, 1e5]
# clf__kernel = ['linear', 'poly','rbf', 'sigmoid']
param_grid = dict(clf__kernel=['sigmoid', 'rbf'],
                  clf__C=[0.001, 0.1, 1, 10, 100, 1000, 10000, 1e3, 5e3, 1e4, 5e4, 1e5],
                  clf__gamma=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                  reduce_dim__n_components=[1, 2, 4, 6, 8, 10, 12])

# grid_search = GridSearchCV(pipe, param_grid=param_grid)
# scoring='%s_macro' % scores[1],
grid_search = GridSearchCV(pipe, param_grid=param_grid, refit=True, cv=10)
grid_search.fit(features_train, labels_train)
labels_predict = grid_search.predict(features_test)

from sklearn.metrics import classification_report

print(classification_report(labels_test, labels_predict))
print(grid_search.best_estimator_.named_steps['reduce_dim'].n_components)
# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(grid_search.best_estimator_, my_dataset, features_list)

# C = 10000  # SVM regularization parameter
# gamma = 1.0
# classifier = SVC(kernel='rbf', C=C, gamma=gamma)
