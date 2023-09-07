from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


# Initialize a new plot
def newplot(width = 8, height = 8, fontsize = 20, style = 'sans-serif', auto_layout = True):


    fig, axes = plt.subplots(figsize=(width,height))

    plt.rcParams['figure.figsize'] = (width,height)
    plt.rcParams['font.family'] = style
    plt.rcParams['figure.autolayout'] = auto_layout
    plt.rcParams['font.size'] = str(fontsize)

    return fig, axes


def add_whitespace(ax = None, upper_fraction = 1.333, lower_fraction = 1):

    # handle defualt axis
    if ax is None:
        ax = plt.gca()

    # check if log scale
    scale_str = ax.get_yaxis().get_scale()

    bottom, top = ax.get_ylim()

    if scale_str == "log":
        upper_fraction = np.power(10, upper_fraction - 1)
        lower_fraction = np.power(10, lower_fraction - 1)
    
    ax.set_ylim([bottom / lower_fraction, top * upper_fraction])



# function to add a stamp to figures
def stamp(left_x, top_y,
          ax=None,
          delta_y=0.05,
          textops_update=None,
          boldfirst = True,
          **kwargs):
    
     # handle defualt axis
    if ax is None:
        ax = plt.gca()
    
    # text options
    textops = {'horizontalalignment': 'left',
               'verticalalignment': 'center',
               'fontsize': 18,
               'transform': ax.transAxes}
    if isinstance(textops_update, dict):
        textops.update(textops_update)
    
    # add text line by line
    for i in range(len(kwargs)):
        y = top_y - i*delta_y
        t = kwargs.get('line_' + str(i))


        if t is not None:
            if boldfirst and i == 0:
                ax.text(left_x, y, t, weight='bold', **textops)
            else:
                ax.text(left_x, y, t, **textops)
