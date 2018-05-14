#!/usr/local/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from sklearn import svm

sys.path.append("../tools/")
from email_preprocess import preprocess

# def make_meshgrid(x, y, h=.02):
#     """Create a mesh of points to plot in
#
#     Parameters
#     ----------
#     x: data to base x-axis meshgrid on
#     y: data to base y-axis meshgrid on
#     h: stepsize for meshgrid, optional
#
#     Returns
#     -------
#     xx, yy : ndarray
#     """
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     return xx, yy
#
#
# def plot_contours(ax, clf, xx, yy, **params):
#     """Plot the decision boundaries for a classifier.
#
#     Parameters
#     ----------
#     ax: matplotlib axes object
#     clf: a classifier
#     xx: meshgrid ndarray
#     yy: meshgrid ndarray
#     params: dictionary of params to pass to contourf, optional
#     """
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
# classifier = svm.SVC(C=1.0, gamma=1.0, kernel='linear')
# classifier.fit(features_train, labels_train)
# classifier.predict(features_test)
# print(classifier.score(features_test, labels_test))

    C = 10000  # SVM regularization parameter
    gamma = 1.0

    # features_train = features_train[:len(features_train) / 100]
    # labels_train = labels_train[:len(labels_train) / 100]

    classifier = svm.SVC(kernel='rbf', C=C, gamma=gamma)

t0 = time()
classifier.fit(features_train, labels_train)
print("training time:", round(time() - t0, 3), "s")

t0 = time()
# score = classifier.score(features_test, labels_test)
predict = classifier.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")
# print("Score", score)
sarah = 0
chris = 0

for p in predict:
    if p == 0:
        sarah = sarah + 1
    else:
        chris = chris + 1

print("Chris:", chris)
# predict1 = classifier.predict(np.array(features_test[10].reshape(1, -1)))
# predict2 = classifier.predict(np.array(features_test[26].reshape(1, -1)))
# predict3 = classifier.predict(np.array(features_test[50].reshape(1, -1)))

# print(predict1[0], labels_test[10])
# print(predict2[0], labels_test[26])
# print(predict3[0], labels_test[50])

# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# ('training time:', 0.0, 's')
# ('prediction time:', 154.024, 's')
# ('Score', '0.9840728100113766')

# subset+linear
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# ('training time:', 0.0, 's')
# ('prediction time:', 0.082, 's')
# ('Score', '0.8845278725824801')

# subset+rbf
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# ('training time:', 0.097, 's')
# ('prediction time:', 0.96, 's')
# ('Score', '0.8890784982935154')

# subset+rbf+c10
# 'Score', '0.9010238907849829')

# subset+rbf+c100
# ('prediction time:', 0.926, 's')
# ('Score', '0.9010238907849829')

# subset+rbf+c100
# ('prediction time:', 0.925, 's')
# ('Score', '0.9010238907849829')

# svm.LinearSVC(C=C),
# svm.SVC(kernel='rbf', gamma=0.7, C=C),
# svm.SVC(kernel='poly', degree=3, C=C))
# models = (svm.SVC(kernel='linear', C=C))
# models = (clf.fit(features_train, labels_train) for clf in models)
# for classifier in models:

# # title for the plots
# titles = ('SVC with linear kernel',
#           'LinearSVC (linear kernel)',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 3) kernel')
#
# # Set-up 2x2 grid for plotting.
# fig, sub = plt.subplots(2, 2)
# plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
# X0, X1 = features_train[:, 0], features_train[:, 1]
# xx, yy = make_meshgrid(X0, X1)
#
# for clf, title, ax in zip(models, titles, sub.flatten()):
#     plot_contours(ax, clf, xx, yy,
#                   cmap=plt.cm.coolwarm, alpha=0.8)
#     ax.scatter(X0, X1, c=features_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel('Sepal length')
#     ax.set_ylabel('Sepal width')
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.set_title(title)

# plt.show()

#########################################################
