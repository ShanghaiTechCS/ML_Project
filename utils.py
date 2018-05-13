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


def classification_fusion(cls_pred, reg_pred, thres_high, thres_low):

    """
    This function calculate the synthesis classification results.
    The final result will take regression result as consideration.

    If the reg_pred[i] > thres_high, we will change cls_pred[i] = 1
    If the reg_pred[i] < thres_low, we will change cls_pred[i] = 0

    """

    synthesis_high = np.where((np.array(reg_pred > thres_high).astype(np.int16) ==1))[0]
    synthesis_low = np.where((np.array(reg_pred < thres_low).astype(np.int16) ==1))[0]

    cls_pred[synthesis_high] = 1
    cls_pred[synthesis_low]  = 0

    return cls_pred