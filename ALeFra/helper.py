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


import os
from shutil import rmtree
import numpy as np
from math import sqrt,floor,ceil
import scipy
from skimage.transform import resize,rotate
from PIL import ImageFont, Image, ImageDraw
from scipy.interpolate import interp1d


#some colors for use in plots
colors = ['red','blue','green', 'cyan', 'magenta', 'yellow','brown','black','pink','orange','white','gray'] + \
                ['red','blue','green', 'cyan', 'magenta', 'yellow','brown','black','pink','orange','white','gray'] + \
                ['red','blue','green', 'cyan', 'magenta', 'yellow','brown','black','pink','orange','white','gray'] + \
                ['red','blue','green', 'cyan', 'magenta', 'yellow','brown','black','pink','orange','white','gray'] + \
                ['red','blue','green', 'cyan', 'magenta', 'yellow','brown','black','pink','orange','white','gray']
markers = ['o', 'v', 'x', '+', '*', '^', '>', '<', 's', 'p', 'h', 'H', 'D', 'd', '1', '2', '3', '4', '8', '.'] + \
                ['o', 'v', 'x', '+', '*', '^', '>', '<', 's', 'p', 'h', 'H', 'D', 'd', '1', '2', '3', '4', '8', '.'] + \
                ['o', 'v', 'x', '+', '*', '^', '>', '<', 's', 'p', 'h', 'H', 'D', 'd', '1', '2', '3', '4', '8', '.'] + \
                ['o', 'v', 'x', '+', '*', '^', '>', '<', 's', 'p', 'h', 'H', 'D', 'd', '1', '2', '3', '4', '8', '.'] + \
                ['o', 'v', 'x', '+', '*', '^', '>', '<', 's', 'p', 'h', 'H', 'D', 'd', '1', '2', '3', '4', '8', '.']

def zero_one_loss(prediction, target):
    if len(prediction) == 0:
        return 0.0
    correct = prediction == target
    return len(np.where(correct)[0]) / len(prediction)

def preprocess_images(images):
    # check whether it is already an image np.ndarray
    if not (isinstance(images, (np.ndarray, np.generic)) and images.ndim > 1):
        images = resize_image_tuple(images,(50,50,3))
    # convert to float image with range 0..1
    images = convert_to_float_images(images)
    # if it is a black and white image, expand dim 3
    if images.ndim == 3:
        images = np.expand_dims(images, 3)
    return images

def convert_probas_to_max_cls_proba(pred):
    if np.array(pred).ndim == 1:
        return pred
    else:
        return pred.max(axis=1)

def merge_images_or_placeholder(samples):
    if len(samples) > 0:
        return merge_images(samples)
    else:
        return annotate_image(np.zeros((100, 100, 3)), 'no sample')

def create_directory_if_not_defined(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def delete_files_in_directory(dir,recursive=False):
    for the_file in os.listdir(dir):
        file_path = os.path.join(dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path) and recursive: rmtree(file_path)
        except Exception as e:
            print(e)

def setup_clean_directory(dir):
    create_directory_if_not_defined(dir)
    delete_files_in_directory(dir,recursive=True)



def annotate_image(img,text,position=(0,0),font_size=(10),font_color=(255, 128, 128)):
    imgshape = img.shape
    n_dims = len(imgshape)
    if n_dims == 1:
        side_length = sqrt(imgshape[0])
        imgshape = (side_length,side_length,3)
        img.reshape((side_length,side_length))
        img = np.stack((img,) * 3,axis=-1)
    elif n_dims == 2 or (n_dims == 3 and imgshape[2] == 1):
        img = np.stack((img.squeeze(),) * 3,axis=-1)
        imgshape = img.shape


    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf", font_size)

    interp = interp1d([img.min(), img.max()+1], [0, 255])

    imgin = interp(img).astype('uint8')

    im1 = Image.fromarray(imgin)
    draw = ImageDraw.Draw(im1)
    draw.text(position, str(text), font_color, font=font)
    return np.array(draw.im,dtype=np.float32).reshape(imgshape)/255

def resize_image_tuple(imgs,size=(300,300,3)):
    img_arr = np.zeros((len(imgs),size[0],size[1],size[2]))
    for i,img in enumerate(imgs):
        img_arr[i] = scipy.misc.imresize(img, size)
    return img_arr

def merge_images(images, size=None):
    if size==None:
        size = (ceil(sqrt(images.shape[0])),ceil(sqrt(images.shape[0])))

    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img

def convert_to_float_images(imgs):
    min = imgs.min()
    max = imgs.max()
    return (imgs-min)/(max-min)

def hstack_images(*imgs, scale_to_max_height=True):
    shapes = np.array([i.shape for i in imgs])
    maxh, maxw, maxc = shapes.max(axis=0)
    if scale_to_max_height:
        imgs = [resize(img,(maxh,int(maxh*(img.shape[1]/img.shape[0])),maxc)) for img in imgs]
        #recalculate this
        shapes = np.array([i.shape for i in imgs])
        maxh, maxw, maxc = shapes.max(axis=0)

    heights = shapes[:, 0]
    widths = shapes[:, 1]
    cum_widths = 0
    sumw = widths.sum()
    collage = np.zeros((maxh,sumw,maxc))
    for img,height,width in zip(imgs,heights,widths):
        h = floor((maxh-height)/2)
        collage[h:h+height,cum_widths:cum_widths+width,:] = img
        cum_widths += width
    return collage

def vstack_images(*imgs, scale_to_max_width=True):
    imgs = [np.rot90(img,axes=(0, 1)) for img in imgs]
    collage = hstack_images(*imgs,scale_to_max_height=scale_to_max_width)
    return np.rot90(collage,axes=(1, 0))

