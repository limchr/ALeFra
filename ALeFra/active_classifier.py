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


import importlib
alefra_spec = importlib.util.find_spec("ALeFra")
if not alefra_spec is not None: # alefra was installed using setup tools
    import sys
    sys.path.append('/media/compute/homes/climberg/src/python/alefra')



import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE,MDS

import ALeFra.helper as helper
from ALeFra.helper import setup_clean_directory, annotate_image, resize_image_tuple, merge_images, convert_to_float_images, hstack_images, vstack_images, markers,colors
import ALeFra.active_strategies as active_strategies



class active_classifier:
    def __init__(self,
                 classifier,
                 x,
                 y=None,
                 imgs=None,
                 fit_method=None,
                 predict_method=None,
                 predict_proba_method=None,
                 score_method=None,
                 incremtenal_trainable=True,
                 classifier_args=(),
                 classifier_kwargs=dict(),
                 fit_args=(),
                 fit_kwargs={}
                 ):
        """Initializes an active classifier from base classifier.

        - **classifier**: The base classifier. Any classifier can be used, just transmit the actual class of the
        classifier and not an instance of it.
        - **fit_method**: The method that can train (fit) the classifier as a callback function.
        - **predict_method**: The predict method of the classifier, should work like classifiers in scikit-learn.
        - **predict_proba_method**: The predict_proba method is used for estimating the posterior probability in query
        strategies like uncertainty sampling.
        - **score_method**: The score method is optional if a different scoreing should be used for visualizing training
        - **x**: The feature vector used for training. This vector is internally saved and a labeled and unlabeled pool
        is created based of this.
        - **y**: The corresponding labels of x
        - **imgs**: The imgs corresponding to samples of feature vector x. Needed to be displayed by visualization.
        - **incremtenal_trainable**: If the classifier is trainable online (incremental). If the classifier is not
        trainable incremental, it is instanciated on each training batch and trained from scratch.
        - **classifier_args**: Positional arguments which are passed to the classifier when it is instanciated.
        - **classifier_kwargs**: Keyword arguments which are passed to the classifier when it is instanciated.
        Should be passed as a dict.
        """
        self.classifier_ = classifier
        self.classifier_args = classifier_args
        self.classifier_kwargs = classifier_kwargs
        self.fit_args = fit_args
        self.fit_kwargs = fit_kwargs
        self.cls = self.classifier_(*self.classifier_args,**self.classifier_kwargs)

        self.fit_ = fit_method if fit_method is not None else self.classifier_.fit
        self.predict_ = predict_method if predict_method is not None else self.classifier_.predict
        self.predict_proba_ = predict_proba_method if predict_proba_method is not None else self.classifier_.predict_proba
        self.score_ = score_method # if score_method is not None else None

        self.incremental_trainable = incremtenal_trainable

        self.epoch_counter_ = 0

        self.init_samples_(x,y,imgs)
        self.set_query_strategy(active_strategies.strategy_query_least_confident)

        self.train_scores_labeled = []
        self.train_scores_unlabeled = []
        self.train_scores_whole = []



    #########################
    #initialization methods
    #########################

    def init_log(self,export_dir,visualization_method='tsne',x_test = None,y_test = None,img_test = None):
        """ Initializes the log capabilities, visualization method (for creating 2d plot) set a test set (optional corresponding images) and define where to save logged data.
        - **export_dir**: The visualization images are stored in subdirectories of the passed directory.
        - **visualization_method**: the visualization method used by feature_visualization method
        - **x_test**: features of test set
        - **y_test**: labels of test set
        - **img_test**: if data set is an image data set, the images can be passed to create nice collages of samples while training
        """
        self.visualization_method = visualization_method

        self.DIR_EXPORT = export_dir
        self.DIR_QUERIED = os.path.join(self.DIR_EXPORT,'log','queried')
        self.DIR_NEW_CORRECT = os.path.join(self.DIR_EXPORT,'log','new_correct')
        self.DIR_NEW_FALSE = os.path.join(self.DIR_EXPORT,'log','new_false')
        self.DIR_LABEL_CHANGE = os.path.join(self.DIR_EXPORT,'log','label_change')
        self.DIR_COLLAGE = os.path.join(self.DIR_EXPORT,'log','collage')
        self.DIR_FEATURE_VISUALIZATION = os.path.join(self.DIR_EXPORT,'log','feature_visualization')
        self.DIR_TRAIN_CLASSIFY = os.path.join(self.DIR_EXPORT,'log','train_classify')


        setup_clean_directory(self.DIR_EXPORT)
        setup_clean_directory(self.DIR_QUERIED)
        setup_clean_directory(self.DIR_NEW_CORRECT)
        setup_clean_directory(self.DIR_NEW_FALSE)
        setup_clean_directory(self.DIR_LABEL_CHANGE)
        setup_clean_directory(self.DIR_COLLAGE)
        setup_clean_directory(self.DIR_FEATURE_VISUALIZATION)
        setup_clean_directory(self.DIR_TRAIN_CLASSIFY)

        if x_test is not None:
            self.set_test_set_(x_test,y_test,img_test)

    def init_samples_(self,x,y,imgs):
        """Sets up the training samples and initializes the labeled / unlabeled pool"""
        self.x_ = np.array(x)
        self.y_ = np.array(y)
        if imgs is not None:
            self.imgs = helper.preprocess_images(imgs)
        else:
            self.imgs = None
        self.labeled_i_ = np.array([],dtype=np.int32)
        self.unlabeled_i_ = np.array(range(len(x)),dtype=np.int32)


    def append_samples(self,x,y):
        """Append samples to the unlabeled pool for incremental learning"""
        former_length = len(self.y_)
        self.x_ = np.vstack((self.x_, x))
        self.y_ = np.hstack((self.y_, y))
        self.unlabeled_i_ = np.hstack((self.unlabeled_i_, list(range(former_length, len(self.y_)))))


    def set_test_set_(self,x_test,y_test,img_test = None):
        """Sets a test set for validating the active learning training procedure. This is necessary in order to call eval_test_set.
        """
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        if img_test is not None:
            self.imgs_test = helper.preprocess_images(img_test)
        else:
            self.imgs_test = None
        self.test_scores = []
        self.test_correct = []
        self.test_false = []

        self.last_test_correct = np.array([False]*len(x_test))
        self.last_test_new_correct_i = np.array([])
        self.last_test_new_false_i = np.array([])
        self.last_test_predicted = np.array([-1]*len(x_test))


    def set_query_strategy(self,strat,*args,**kwargs):
        """Sets a querying strategy that the active learning should use.

        - **strat**: Callback function that takes at least 2 arguments:
            1. Instance of the active_classifier class
            2. batch size (how many samples should be queryied by the strategy?)
        - **query_args**: other arguments that are passed to the query function
        - **query_kwargs**: keyword arguments that are passed to the query function
        """
        self.query_strategy = strat
        self.query_args = args
        self.query_kwargs = kwargs




    #####################################
    #getter-setter
    #####################################

    def get_unlabeled_x(self):
        return self.x_[self.unlabeled_i_]
    def get_labeled_x(self):
        return self.x_[self.labeled_i_]
    def get_unlabeled_y(self):
        return self.y_[self.unlabeled_i_]
    def get_labeled_y(self):
        return self.y_[self.labeled_i_]


    def get_train_scores(self):
        return (self.train_scores_whole,self.train_scores_labeled,self.train_scores_unlabeled)
    def get_test_scores(self):
        return self.test_scores












    def update_labels_(self,unlabeled_pool_indexes):
        """Updates the labeled and unlabeled pool: puts samples from the labeled to unlabeled pool
        -**unlabeled_pool_indexes**: indices of samples from unlabeled pool to put in labeled pool (because they have been labeled and trained)
        """

        indexes = self.unlabeled_i_[unlabeled_pool_indexes]
        self.unlabeled_i_ = np.delete(self.unlabeled_i_,unlabeled_pool_indexes,axis=0)
        self.labeled_i_ = np.append(self.labeled_i_,indexes)
        return indexes

    def fit_active(self,batch_size):
        """Use this function to train the classifier. The functions calls the querying function to query the best samples,
        then those samples gets labeled automatically and the labeled and unlabeled pool are adjusted.
        """

        i = self.query(batch_size)
        if len(i) == 0:
            print('no sample queried')
            return False

        indices = self.fit_unlabeled_indices(i)
        return indices

    def query(self,batch_size):
        """calls the querying method and returns the next to be labeled samples."""
        if len(self.unlabeled_i_) == 0:
            print('there is no sample in unlabeled pool to train')
            return False
        i = self.query_strategy(self,batch_size,*self.query_args,**self.query_kwargs)
        return i

    def fit_unlabeled_indices(self,i):
        """Fit classifier either incremental or from scratch with passed indices from unlabeled pool. Returns indices of whole data set of newly labeled samples."""
        if self.incremental_trainable:
            self.fit_(self.cls,self.get_unlabeled_x()[i],self.get_unlabeled_y()[i],*self.fit_args,**self.fit_kwargs)
            indices = self.update_labels_(i)
        else:
            """for classifiers that are not able to train incrementally/online, reinitialize a new classifier and train from scratch with all yet labeled samples"""
            indices = self.update_labels_(i)
            self.cls = self.classifier_(*self.classifier_args, **self.classifier_kwargs)
            try:
                self.fit_(self.cls,self.get_labeled_x(),self.get_labeled_y(), *self.fit_args, **self.fit_kwargs)
            except:
                print('unable to train, maybe not enough labeled samples')
                # print(traceback.format_exc())

        # this is maybe a bit dirty
        self.last_queried_i_ = indices
        self.epoch_counter_ += 1

        return indices



    def evaluate(self,score_train_set=True,log_collage_image=True,log_single_images=True):
        """Evaluate the train and test set and creates images of the progress and accuracy. Call this function after *fit_active* to monitor
        the progress of the active learning training. Images are saved to output dir, that you can change in the *__init__*.
        - **score_train_set**: Also score train set.
        - **log_single_images**: if True, single images are saved (one image for each batch which shows: queried samples,
        new correct samples, new false sampled and samples which label was changed)
        - **log_collage_image**: if True, a collage image of single images from above is saved.
        """

        if score_train_set:
            try:
                predicted_x = np.array(self.predict_(self.cls,self.x_))
                self.train_scores_labeled.append(self.calc_score_(predicted_x[self.labeled_i_],self.get_labeled_y()))
                self.train_scores_unlabeled.append(self.calc_score_(predicted_x[self.unlabeled_i_],self.get_unlabeled_y()))
                self.train_scores_whole.append(self.calc_score_(predicted_x,self.y_))
            except:
                print('unable to score train set')
                print(traceback.format_exc())



        try:
            self.test_predicted = np.array(self.predict_(self.cls,self.x_test))
        except:
            print('error in predict, cant eval_test_set')
            return False
        self.test_correct = (self.y_test == self.test_predicted)
        new_correct = np.logical_and(self.test_correct, np.logical_not(self.last_test_correct))
        self.new_correct_i = np.where(new_correct)[0]
        new_false = np.logical_and(self.last_test_correct, np.logical_not(self.test_correct))
        self.new_false_i = np.where(new_false)[0]
        label_change = self.test_predicted != self.last_test_predicted
        self.label_change_i = np.where(label_change)[0]

        test_score = self.calc_score_(self.y_test, self.test_predicted)
        self.test_scores.append(test_score)


        if self.imgs_test is not None:
            if log_single_images or log_collage_image:
                # save the images queried for the last batch
                img_last_queried = helper.merge_images_or_placeholder(self.imgs[self.last_queried_i_])
                # save the images that are correct for the last batch and that weren't correct in the batch before
                img_new_correct = helper.merge_images_or_placeholder(self.imgs_test[self.new_correct_i])
                # save the images that were correct before and since the last batch are not correct anymore
                img_new_false = helper.merge_images_or_placeholder(self.imgs_test[self.new_false_i])
                # save the images which label did changed since last batch (could also be wrong label to wrong label)
                img_label_change = helper.merge_images_or_placeholder(self.imgs_test[self.label_change_i])

            # log single images
            if log_single_images:
                self.log_image(img_last_queried,self.DIR_QUERIED)
                self.log_image(img_new_correct,self.DIR_NEW_CORRECT)
                self.log_image(img_new_false,self.DIR_NEW_FALSE)
                self.log_image(img_label_change,self.DIR_LABEL_CHANGE)
            # log nicely collage image
            if log_collage_image:
                queried_txt = annotate_image(np.zeros((10,200,3)),text='queried samples')
                correct_txt = annotate_image(np.zeros((10,200,3)),text='new correct samples')
                false_txt = annotate_image(np.zeros((10,200,3)),text='new false samples')
                label_changed_txt = annotate_image(np.zeros((10,200,3)),text='label changed samples')

                collage = hstack_images(vstack_images(queried_txt,img_last_queried),vstack_images(correct_txt,img_new_correct),vstack_images(false_txt,img_new_false),vstack_images(label_changed_txt,img_label_change),scale_to_max_height=True)
                try:
                    collage = vstack_images(collage,annotate_image(np.zeros((20,200,3)),('Accurracy: %f, Gain: %f'%(self.test_scores[-1],self.test_scores[-1]-self.test_scores[-2]))),scale_to_max_width=True)
                except:
                    pass
                self.log_image(collage,self.DIR_COLLAGE)
        # save variables from this batch to the variables related to the last batch
        self.last_test_correct = self.test_correct
        self.last_test_new_correct_i = self.new_correct_i
        self.last_test_new_false_i = self.new_false_i
        self.last_test_predicted = self.test_predicted



    def visualize_by_embedding(self):
        """Is called e.g. after each trained batch (fit_active) to create a 2d-visualization of the training progress. The visualization
        embedding is created at the first call to visualize_by_embedding and may take a couple of minutes to calculate (depends on the size of the data set).
        """
        plt.cla()
        plt.ion()
        ax = plt.gca()

        # do dimension reduction
        if self.x_.shape[1] > 2:
            if self.visualization_method == 'tsne':
                if not hasattr(self,'tsne'):
                    print('calculating tsne visualization')
                    self.tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
                                       n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
                                       init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)
                    self.x_visu = self.tsne.fit_transform(self.x_,self.y_)
            elif self.visualization_method == 'mds':
                if not hasattr(self, 'mds'):
                    print('calculating mds visualization')
                    self.mds = MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001,
                                   n_jobs=1, random_state=None, dissimilarity='euclidean')
                    self.x_visu = self.mds.fit_transform(self.x_, self.y_)
            ux, uy, lx, ly = self.x_visu[self.unlabeled_i_], self.get_unlabeled_y(), self.x_visu[self.labeled_i_], self.get_labeled_y()
        else:
            ux, uy, lx, ly = self.get_unlabeled_x(), self.get_unlabeled_y(), self.get_labeled_x(), self.get_labeled_y()

        for x, y in zip(ux,uy):
            plt.scatter(x[0], x[1], marker=markers[y], color='grey')  # some_colors[int(y)]
        for i, (x, y) in enumerate(zip(lx,ly)):
            plt.scatter(x[0], x[1], marker=markers[y], color=colors[y])  # some_colors[int(y)]
            ax.annotate(i, (x[0],x[1]))
        try:
            plt.title('Feature Visualization. Train-score: %5f, test-score: %5f' % (self.train_scores_whole[-1],self.test_scores[-1]))
        except:
            pass

        plt.savefig(os.path.join(self.DIR_FEATURE_VISUALIZATION, str(self.epoch_counter_).zfill(8) + '.png'), format='png')




        plt.cla()
        plt.ion()
        ax = plt.gca()

        predicted_x = np.array(self.predict_(self.cls, self.x_))

        for x,y,predicted in zip(self.x_, self.y_, predicted_x):
            plt.scatter(x[0], x[1], marker=markers[y], color=colors[predicted])  # some_colors[int(y)]

        plt.savefig(os.path.join(self.DIR_TRAIN_CLASSIFY, str(self.epoch_counter_).zfill(8) + '.png'), format='png')


    def save_scores_plot(self):
        """Saves a score plot (test and train accuracies).
        """
        plt.cla()
        plt.ion()

        plt.title('Active learning scores. Final Test-Accuracy: '+str(self.test_scores[-1])+', Final Train-Accuracy: '+str(self.train_scores_whole[-1]))
        plt.plot(self.test_scores,label='Test Accuracy',c='red')
        plt.plot(self.train_scores_whole,label='Train Accuracy',c='blue')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(os.path.join(self.DIR_EXPORT, 'scores.png'), format='png')







    ###############################
    #helper
    ###############################

    def calc_score_(self,prediction,target):
        if self.score_ is not None:
            return self.score_(self.cls,prediction,target)
        else:
            return helper.zero_one_loss(prediction,target)

    def log_image(self,img,dir):
        plt.imsave(os.path.join(dir, (str(self.epoch_counter_).zfill(8)) + '.png'), img.squeeze())

    def exec_cls_fun(self,fun,*args,**kwargs):
        if isinstance(fun,property):
            return fun.__get__(self.cls)(*args, **kwargs)
        else:
            return fun.__get__(self.cls)(*args, **kwargs)
