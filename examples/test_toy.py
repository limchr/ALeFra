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

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# import the main functionality of ALeFra
from ALeFra.active_classifier import active_classifier
# import some predefined querying strategies
from ALeFra.active_strategies import strategy_query_random, strategy_query_least_confident, strategy_query_lower_percentile_randomly



def load_toy_4(n,overlap):
    """creates a toy data set with 4 classes and a specific overlap"""
    cl1_x = np.array([np.random.random(n), np.random.random(n)]).T
    cl1_y = np.array([0] * n)
    cl2_x = np.array([np.random.random(n)+(1-overlap), np.random.random(n)]).T
    cl2_y = np.array([1] * n)
    cl3_x = np.array([np.random.random(n), np.random.random(n)+(1-overlap)]).T
    cl3_y = np.array([2] * n)
    cl4_x = np.array([np.random.random(n)+(1-overlap), np.random.random(n)+(1-overlap)]).T
    cl4_y = np.array([3] * n)
    x = np.vstack((cl1_x, cl2_x,cl3_x,cl4_x))
    y = np.hstack((cl1_y,cl2_y,cl3_y,cl4_y))

    return (x,y)


# load data set
x, y = load_toy_4(200,0)
# doing a train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#train normal SVM and calc score on test set
a = SVC(kernel='linear')
a.fit(x_train,y_train)
print('normal svm score: ',a.score(x_test,y_test))

# export dir for saving plots
EXPORT_DIR = 'export_toy'
# to do uncertainty sampling we need certainty information from the SVM (SVC scikit-learn classifier)
svm_kwargs = {'probability':True}

#convert scikit-learn SVC classifier to an active classifier using ALeFra
active_cls = active_classifier(classifier=SVC,  # support vector classifier from scikit-learn
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
active_cls.set_query_strategy(strategy_query_random)

#set test set
active_cls.init_log(EXPORT_DIR,visualization_method='tsne',x_test = x_test,y_test = y_test,img_test = None)

#active querying and testing/logging
for epoch in range(100):
    print(epoch)
    active_cls.fit_active(1)  # query batch of size 1 and train it
    active_cls.evaluate()  # evaluate test and train set
    active_cls.visualize_by_embedding()  # visualize training progress using a t-SNE embedding (saved in export dir)


# save a plot of scores while training
active_cls.save_scores_plot()
