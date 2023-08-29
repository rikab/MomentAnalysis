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


def stamp():
    pass