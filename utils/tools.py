# version 2.1 du 21 mai 2022

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#import pandas as pd
#from seaborn import heatmap
#from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

def split_stratified_into_train_val_test(dataset: (ndarray, ndarray), 
                                         frac_train:float=0.7, 
                                         frac_val:float=0.15,
                                         frac_test:float=0.15, 
                                         shuffle:bool=False,
                                         seed:float=None): 
    '''
    Splits a numpy dataset (data, labesl) into three subsets train, val, and test
    following fractional ratios provided, where each subset is stratified by the labels array. 
    Splitting into 3 datasets is achieved by running train_test_split() twice.

    Parameters
    ----------
    dataset: the dataset to split, as a tuple (data, labels)
    frac_train, frac_val, frac_test  : The ratios with which the dataset will be split into 
             train, val, and test data. The values should be expressed as float fractions 
             and should sum to 1.0.
    shuffle: wether to suffle the datset before splitting or not.
    seed: the seed to pass to train_test_split().

    Returns
    -------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) : the 3 sub-dataset.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))
    X, Y = dataset

    # note on train_test_split : Stratified train/test split is not implemented for shuffle=False 

    # Split the dataset into train and temp datasets:
    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, 
                                                        stratify=Y, 
                                                        test_size=1.0-frac_train,
                                                        shuffle=True,
                                                        random_state=seed)
                                                        
    # Split the temp datafset into val and test datasets:
    relative_frac_test = frac_test / (frac_val + frac_test)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                    stratify=y_temp,
                                                    test_size=relative_frac_test,
                                                    shuffle=True,
                                                    random_state=seed)
    
    assert len(X) == len(x_train) + len(x_val) + len(x_test)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)



def plot_loss_accuracy(hist:list, min_acc=None, max_loss=None, training=True, single_color:bool = True):
    '''Plot training & validation loss & accuracy values, giving an argument
       'hist' of type 'tensorflow.python.keras.callbacks.History'. '''
    
    custom_lines = [Line2D([0], [0], color='blue', lw=1, marker='o'),
                    Line2D([0], [0], color='orange', lw=1, marker='o')]
    val_colors = ('lightcoral', 'orange', 'gold', 'goldenrod', 'darkgoldenrod')
    
    plt.figure(figsize=(15,5))
    
    if not isinstance(hist, list): hist = [hist]
        
    ax1 = plt.subplot(1,2,1)
    for (i, h) in enumerate(hist):
        if h.history.get('accuracy') and training:
            ax1.plot(np.array(h.epoch)+1, h.history['accuracy'], 'o-', 
                     markersize=1,
                     color='blue', label=f'train-{i:01d}')
        if h.history.get('val_accuracy'):
            if single_color:
                val_color = 'orange'
            else:
                val_color = val_colors[i % len(val_colors)]
            ax1.plot(np.array(h.epoch)+1, h.history['val_accuracy'], 'o-', 
                     markersize=1,
                     color=val_color, label=f'val-{i:01d}')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch') 
    y_min, y_max = ax1.get_ylim()
    if min_acc is not None:
        ax1.set_ylim((min_acc, y_max))
    #ax1.set_xticks(np.arange(1, len(h.epoch)+1))
    ax1.grid(which='major', color='xkcd:cool grey',  linestyle='-',  alpha=0.7)
    ax1.grid(which='minor', color='xkcd:light grey', linestyle='--', alpha=0.5)
    ax1.legend()
    #ax1.legend(custom_lines, ['Train', 'Valid'])
    
    # Plot training & validation loss values
    ax2 = plt.subplot(1,2,2)
    for (i, h) in enumerate(hist):
        if h.history.get('loss') and training:
            ax2.plot(np.array(h.epoch)+1, h.history['loss'], 'o-',  
                     markersize=1,
                     color='blue', label=f'train-{i:01d}')
        if h.history.get('val_loss'):
            if single_color:
                val_color = 'orange'
            else:
                val_color = val_colors[i % len(val_colors)]
            ax2.plot(np.array(h.epoch)+1, h.history['val_loss'], 'o-', 
                     markersize=1,
                     color=val_color, label=f'val-{i:01d}')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    if max_loss is not None:
        ax2.set_ylim((0, max_loss))

    #ax2.set_xticks(np.arange(1, len(h.epoch)+1))
    ax2.grid(which='major', color='xkcd:cool grey',  linestyle='-',  alpha=0.7)
    ax2.grid(which='minor', color='xkcd:light grey', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.show()

def plot_images(image_array:np.ndarray, R:int, C:int, r:int=0, figsize:tuple=None, reverse:bool=False):
    '''
    Plot the images from image_array on a R x C grid, starting at image rank r.
    Arguments:
       image_array: an array of images
       R: the number of rows
       C: the number of columns
       r: the starting rank in the array image_array (default: 0)
       figsize: the sise of the display (default: (C//2+1, R//2+1))
       reverse: wether to reverse video the image or not (default: False)
    '''
    if figsize is None: figsize=(C//2+1, R//2+1)
    plt.figure(figsize=figsize)
    for i in range(R*C):
        plt.subplot(R, C, i+1)
        im = image_array[r+i]
        if reverse: im = 255 - im
        plt.imshow(im, cmap='gray')
        plt.axis('off');


# def show_cm(true, results, classes):
#     ''' true  : the actual labels 
#         results : the labels computed by the trained network (one-hot format)
#         classes : list of possible label values'''
#     predicted = np.argmax(results, axis=-1) # tableau d'entiers entre 0 et 9 
#     cm = confusion_matrix(true, predicted)
#     df_cm = pd.DataFrame(cm, index=classes, columns=classes)
#     plt.figure(figsize=(11,9))
#     heatmap(df_cm, annot=True, cbar=False, fmt="3d")
#     plt.xlabel('predicted label')
#     plt.ylabel('true label')
#     plt.show()
    
def scan_dir(path):
    tree = ''
    data = [item for item in os.walk(path)]
    for item in data:
        if item[2]:
            for file in item[2]:
                tree += f'{item[0]}/{file}\n'
        else:
            tree += f'{item[0]}/\n'
    return tree
    