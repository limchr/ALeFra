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

def strategy_query_random(obj, batch_size):
    return random.sample(range(len(obj.unlabeled_i_)), batch_size)


def strategy_query_least_confident(obj, batch_size):
    """active querying wich takes the batch_size least confident samples. The passed probability is evaluated with the
    *predict_proba* callback, which was passed to *__init__*.
    """
    try:
        pred = obj.predict_proba_(obj.cls, obj.get_unlabeled_x())
    except:
        print('can not predict probability, query randomly')
        return strategy_query_random(obj, batch_size)

    pred = convert_probas_to_max_cls_proba(pred)
    if np.array(pred).all() == 0:
        print('having no prototypes in, so choosing random samples first')
        return strategy_query_random(batch_size)
    else:
        inds = np.argsort(pred)[:batch_size]
        return inds


def strategy_query_lower_percentile_randomly(obj,batch_size,lower_percentile_size=0.2):
    """active querying which queries a random subset of the x% percentile of least confident samples"""

    try:
        pred = obj.exec_cls_fun(obj.predict_proba_,obj.get_unlabeled_x())
    except:
        return strategy_query_random(obj,batch_size)

    pred = helper.convert_probas_to_max_cls_proba(pred)

    if np.array(pred).all() == 0:
        print('having no prototypes in, so choosing random samples first')
        inds = np.random.choice(len(pred), batch_size)
    else:
        percentile_size = int(max(len(pred)*lower_percentile_size,batch_size))
        inds = np.argsort(pred)[:percentile_size]
        try:
            inds = np.random.choice(inds, batch_size)  # ,replace=False)
        except Exception:
            return []
    return inds
