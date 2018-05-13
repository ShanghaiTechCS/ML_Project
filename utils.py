import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_figure(train_data, val_data, start, stop, num_point, legends, xlabels, ylabels, save_path):
    """
    Plot the training and validation figure in this files.
    """

    x = np.linspace(start=start,stop=stop, num=num_point)
    plt.figure()
    plt.plot(x, train_data)
    plt.plot(x, val_data)
    plt.legend(legends)
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.savefig(save_path)


def pure_profit(cls_pred, cls_gt, profit_gt):

    """
    profit_gt: the profit of of every marketing
    cls_pred: the classification result of the models
    cls_gt: the groundtruth classification
    """

    pure_profit_gt = profit_gt - 30
    one_index = np.where((cls_pred==1))[0]
    pure_profit_variable = np.sum(pure_profit_gt[one_index])

    return pure_profit_variable