#!/usr/bin/env python
#
# Copyright (C) 2018
# Christian Limberg
# Centre of Excellence Cognitive Interaction Technology (CITEC)
# Bielefeld University
#
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ALeFra.active_classifier import active_classifier
import ALeFra.active_strategies as active_strategies


# load data set
x, y = sklearn.datasets.load_iris(return_X_y=True)

# doing a train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40, shuffle=True)

# export dir for saving plots
EXPORT_DIR = 'export'

# to do uncertainty sampling we need certainty information from the SVM (SVC scikit-learn classifier)
svm_kwargs = {'probability':True}

#convert scikit-learn SVC classifier to an active classifier using ALeFra
active_cls = active_classifier(classifier=SVC,
                               fit_method=SVC.fit,
                               predict_method=SVC.predict,
                               predict_proba_method=SVC.predict_proba,
                               score_method=None,  # using zero-one-loss as default loss function for plots
                               x = x_train,
                               y = y_train,
                               # True if using an online/incremental trainable classifier,
                               # otherwise the classifier is trained from scratch after each batch
                               incremtenal_trainable=False,
                               classifier_args=(),
                               classifier_kwargs=svm_kwargs,  # Pass keyword arguments for the SVC classifier
        )

# set strategy for querying
active_cls.set_query_strategy(active_strategies.strategy_query_least_confident)

# set test set
active_cls.init_log(EXPORT_DIR,visualization_method='tsne',x_test = x_test,y_test = y_test,img_test = None)

#active learning and testing/logging loop
for epoch in range(100):
    print(epoch)
    active_cls.fit_active(1) # query batch of size 1 and train it
    active_cls.evaluate(score_train_set=True) # evaluate test and train set
    active_cls.visualize_by_embedding() # visualize training progress using a t-SNE embedding (saved in export dir)


# save a plot of scores while training
active_cls.save_scores_plot()


# alternatively a score plot can be calculated manually
train_scores, train_scores_labeled, train_scores_unlabeled = active_cls.get_train_scores()
test_scores = active_cls.get_test_scores()
plt.ion()
plt.plot(train_scores,label='train score')
plt.plot(train_scores_labeled,label='train score of labeled samples')
plt.plot(train_scores_unlabeled,label='train score of unlabeled samples')
plt.plot(test_scores,label='test score')
plt.legend()
plt.xlabel('trained batch')
plt.ylabel('accuracy')
plt.ioff()
plt.show()
