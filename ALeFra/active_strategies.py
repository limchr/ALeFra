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
import random
from ALeFra.helper import convert_probas_to_max_cls_proba
import ALeFra.helper as helper
import numpy as np
import random

from sklearn.linear_model import SGDClassifier
from PracticalMachineLearning.glvq import glvq
from common.classification import get_max_class_probas
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def strategy_query_random(obj, batch_size):
    return random.sample(range(len(obj.unlabeled_i_)), batch_size)


def strategy_query_least_confident(obj, batch_size):
    """active querying wich takes the batch_size least confident samples. The passed probability is evaluated with the
    *predict_proba* callback, which was passed to *__init__*.
    """
    try:
        pred = obj.exec_cls_fun(obj.predict_proba_, obj.get_unlabeled_x())
    except:
        print('can not predict probability: query randomly')
        return strategy_query_random(obj, batch_size)

    pred = convert_probas_to_max_cls_proba(pred)
    if np.array(pred).all() == 0:
        print('can not predict probabilities, it seems there was something wrong: query randomly')
        return strategy_query_random(obj, batch_size)
    else:
        inds = np.argsort(pred)[:batch_size]
        return inds

def strategy_query_least_margin(obj, batch_size):
    # todo
    return None



def strategy_query_by_committee(obj, batch_size):
    class qbc():
        '''class to be injected into active learning model for storing stuff for qbc'''
        def __init__(self):
            self.svm = SVC(kernel='linear', probability=True)
            # self.perceptron = SGDClassifier(loss='perceptron')
            self.logistic_regression = SGDClassifier(loss='log')
            self.generalized_lvq = glvq()
            # self.xgb = XGBClassifier()

        def fit(self, x, y):
            if hasattr(self, 'x'):
                self.x = np.vstack((self.x, x))
                self.y = np.hstack((self.y, y))
            else:
                self.x = x
                self.y = y

            try:
                #train incremental models
                self.logistic_regression.partial_fit(x, y, classes=self.unique_classes)
                self.generalized_lvq.fit(x, y)

                #train offline models
                self.svm = SVC(kernel='linear', probability=True)
                self.svm.fit(self.x, self.y)

                self.tree = DecisionTreeClassifier()
                self.tree.fit(self.x, self.y)

            except:
                print('ERROR: can not train qbc classifiers')

        def query(self, unlabeled_pool,batch_size):
            try:
                probas = np.zeros((0,len(unlabeled_pool)))
                probas_svm = self.svm.predict_proba(unlabeled_pool)
                probas_svm = get_max_class_probas(probas_svm)

                probas_tree = self.tree.predict_proba(unlabeled_pool)
                probas_tree = get_max_class_probas(probas_tree)


                # probas_perceptron = self.perceptron.predict_proba(unlabeled_x_train)
                probas_logistic_regression = self.logistic_regression.predict_proba(unlabeled_pool)
                probas_logistic_regression = get_max_class_probas(probas_logistic_regression)

                probas_generalized_lvq = self.generalized_lvq.predict_proba(unlabeled_pool)


                probas = np.vstack((probas, probas_svm))
                probas = np.vstack((probas, probas_tree))
                probas = np.vstack((probas, probas_logistic_regression))
                probas = np.vstack((probas, probas_generalized_lvq))


                min_i = np.argsort(probas.sum(axis=0))



            except:
                return None
            return min_i[:batch_size]

    if not hasattr(obj,'qbc'):
        obj.qbc = qbc()
        obj.qbc.unique_classes = np.unique(obj.get_unlabeled_y())

    queried = obj.qbc.query(obj.get_unlabeled_x(),batch_size)

    if queried is None:
        print('can not query by commitee, query random samples')
        queried = strategy_query_random(obj, batch_size)

    # fit models in qbc with new queried samples
    obj.qbc.fit(obj.get_unlabeled_x()[queried],obj.get_unlabeled_y()[queried])

    return queried



def strategy_query_lower_percentile_randomly(obj,batch_size,lower_percentile_size=0.2):
    """active querying which queries a random subset of the x% percentile of least confident samples"""
    # try to predict the accuracy of the unlabeled set for uncertainty sampling
    try:
        pred = obj.exec_cls_fun(obj.predict_proba_,obj.get_unlabeled_x())
    # if it is not possible (because e.g. the classifier was not trained yet), then do a random sampling
    except:
        print('can not predict probability: query randomly')
        return strategy_query_random(obj,batch_size)
    # we only want the maximum class probability of all samples
    pred = helper.convert_probas_to_max_cls_proba(pred)
    if np.array(pred).all() == 0:
        print('can not predict probabilities, it seems there was something wrong: query randomly')
        inds = np.random.choice(len(pred), batch_size)
    else:
        percentile_size = int(max(len(pred)*lower_percentile_size,batch_size))
        inds = np.argsort(pred)[:percentile_size]
        try:
            inds = np.random.choice(inds, batch_size, replace=False)
        except Exception:
            return []
    return inds


