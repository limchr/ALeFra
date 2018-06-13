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


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.datasets
from sklearn.decomposition import PCA

from ALeFra.active_classifier import active_classifier
import ALeFra.active_strategies as active_strategies


# load data set
images_1d, y = sklearn.datasets.load_digits(n_class=5, return_X_y=True)
images = images_1d.reshape(-1,8,8)


# # visualize a sample just to make sure data is meaningful
# plt.imshow(images[0].reshape(8,8))
# plt.show()

# extract features using a principle component analysis
pca = PCA(n_components=50)
# x = pca.fit_transform(images_1d)
x = images_1d

# doing a train/test split
x_train, x_test, y_train, y_test, images_train, images_test = train_test_split(x, y, images, test_size=0.2, random_state=42, shuffle=True)


# EXPORT_DIR = '/hri/storage/user/climberg/experiments/interactive_gan/digits_test/'
EXPORT_DIR = 'export'
svm_kwargs = {'probability':True}






#convert glvq to an active classifier using ActiVisuFra
active_cls = active_classifier(classifier=SVC,
                               fit_method=SVC.fit,
                               predict_method=SVC.predict,
                               predict_proba_method=SVC.predict_proba,
                               score_method=None,
                               x = x_train,
                               y = y_train,
                               imgs=images_train,
                               incremtenal_trainable=False,
                               classifier_args=(),
                               classifier_kwargs=svm_kwargs,
        )

# set strategy for querying
active_cls.set_query_strategy(active_strategies.strategy_query_random)

#set test set
active_cls.init_log(EXPORT_DIR,visualization_method='tsne',x_test = x_test,y_test = y_test,img_test = images_test)

#active querying and testing/logging
for epoch in range(200):
    print(epoch)
    r = active_cls.fit_active(5)
    if r is not False and len(active_cls.unlabeled_i_) != 0:
        active_cls.evaluate(score_train_set=True, log_collage_image=True, log_single_images=True)
        active_cls.visualize_by_embedding()


train_scores, train_scores_labeled, train_scores_unlabeled = active_cls.get_train_scores()
test_scores = active_cls.get_test_scores()

plt.clf()
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
