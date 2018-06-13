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


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from common.load_data import load_toy_4
from PracticalMachineLearning.glvq import glvq
import numpy as np
import random

from ALeFra.active_classifier import active_classifier
from ALeFra.active_strategies import strategy_query_random


def strategy_query_voronoi_edges(obj, batch_size, lower_percentile_size=0.2):
    num_prototypes = 3
    ves_values = []
    try:
        dists = obj.cls.dist(obj.get_unlabeled_x(), obj.cls.prototypes)
        for d in dists:
            win_loose_protos = obj.cls.get_win_loose_prototypes(d,num_prototypes)
            ves_score = np.var(d[win_loose_protos])*np.mean(d[win_loose_protos])
            ves_values.append(ves_score)
            print(obj.cls.labels[win_loose_protos],ves_score)
        return np.argsort(np.array(ves_values))[:batch_size]
    except:
        return strategy_query_random(obj,batch_size)



def strategy_query_voronoi_edges_improved(obj, batch_size, lower_percentile_size=0.2):
    num_prototypes = [2,3,4]
    least_confident = [10,10,5]
    values = dict()
    try:
        dists = obj.cls.dist(obj.get_unlabeled_x(), obj.cls.prototypes)
        for d in dists:
            win_loose_protos = obj.cls.get_win_loose_prototypes(d,max(num_prototypes))
            for n in num_prototypes:
                protos = d[win_loose_protos[:num_prototypes]]
                values[n].append(np.var(protos)*np.mean(protos))
        joint_rtn = []
        for i,n in enumerate(num_prototypes):
            least_n = least_confident[i]
            joint_rtn.extend(values[n].sort()[:least_n])
        return random.sample(joint_rtn,batch_size)
    except:
        return strategy_query_random(obj,batch_size)






# load data set
x, y = load_toy_4(200,0)
# doing a train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#train normal SVM
a = SVC(kernel='linear')
a.fit(x_train,y_train)
print('normal svm score: ',a.score(x_test,y_test))


EXPORT_DIR = '/hri/storage/user/climberg/experiments/interactive_gan/toy_ves/'
# EXPORT_DIR = 'export'
# svm_kwargs = {'probability':True}
glvq_kwargs = {'max_prototypes_per_class': 20, 'learning_rate': 2, 'strech_factor': 10}

#convert glvq to an active classifier using ActiVisuFra
active_cls = active_classifier(classifier=glvq,
                               score_method=None,
                               x = x_train,
                               y = y_train,
                               incremtenal_trainable=True,
                               classifier_args=(),
                               classifier_kwargs=glvq_kwargs,
        )

# set strategy for querying
active_cls.set_query_strategy(strategy_query_voronoi_edges_improved)

#set test set
active_cls.init_log(EXPORT_DIR,visualization_method='tsne',x_test = x_test,y_test = y_test,img_test = None)

#active querying and testing/logging
for epoch in range(100):
    print(epoch)
    active_cls.fit_active(1)
    active_cls.evaluate()
    active_cls.visualize_by_embedding()

active_cls.save_scores_plot()
