#!/usr/bin/env python

""" Quickhack , comparing classifier speed performance with python and scikit-learn """

from sklearn import linear_model
from sklearn import naive_bayes
import numpy as np
import time
import random

nattributes = 100
learn_nrows = 1000
score_nrows = 1000000
train = np.array([np.random.rand(nattributes) for i in xrange(learn_nrows)])
target = np.array([random.choice([0,1]) for i in xrange(learn_nrows)])
X,y = train, target

logreg = linear_model.LogisticRegression()
logreg.fit(X,y)

gb = naive_bayes.GaussianNB()
gb.fit(X,y)

deploy = np.array([np.random.rand(nattributes) for i in xrange(score_nrows)])

s1 = time.time()
scores_logreg = [ logreg.predict_proba(i)[:,1][0] for i in deploy]

print "logistic regression: dim={}rows/{}attributes,:{} sec.".format(score_nrows,nattributes,round((time.time() - s1),4))

s2 = time.time()
scores_gb = [ gb.predict_proba(i)[:,1][0] for i in deploy]
print "gaussian naive bayes: dim={}rows/{}attributes,:{} sec.".format(score_nrows,nattributes,round((time.time() - s2),4))

