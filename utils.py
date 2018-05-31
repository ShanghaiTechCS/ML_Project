import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd


def plot_figure(training, validation, start, stop, num_point, name=None, ylabel=None, legend=None):
    x = np.linspace(start=start, stop=stop, num=num_point)
    plt.figure()
    plt.plot(x, training)
    plt.plot(x, validation)
    plt.xlabel('regularization')
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.savefig(name)


def calculate_profit(val_gt_profit, val_cls_pred, val_cls_gt):
    """
    to calculate the profit by the cls regression result

    Args:
        val_gt_profit: the pure profit of a marketing
        val_cls_pred: the cls prediction results.
    """

    predict_true = np.where((val_cls_pred == 1))
    profit = np.sum(val_gt_profit[predict_true])

    return profit


def plot_figure(train_data, val_data, start, stop, num_point, legends, xlabels, ylabels, save_path):
    """
    Plot the training and validation figure in this files.
    """

    x = np.linspace(start=start, stop=stop, num=num_point)
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
    one_index = np.where((cls_pred == 1))[0]
    pure_profit_variable = np.sum(pure_profit_gt[one_index])

    return pure_profit_variable


def classification_fusion(cls_pred, reg_pred, thres_high, thres_low):
    """
    This function calculate the synthesis classification results.
    The final result will take regression result as consideration.

    If the reg_pred[i] > thres_high, we will change cls_pred[i] = 1
    If the reg_pred[i] < thres_low, we will change cls_pred[i] = 0

    """

    synthesis_high = np.where((np.array(reg_pred > thres_high).astype(np.int16) == 1))[0]
    synthesis_low = np.where((np.array(reg_pred < thres_low).astype(np.int16) == 1))[0]

    cls_pred[synthesis_high] = 1
    cls_pred[synthesis_low] = 0

    return cls_pred


def recall_cls(cls_pred, cls_gt):
    """
    To compute th recall
    :param cls_pred: (N,)
    :param cls_gt:  (N,)
    :return:
    """

    total_recall = np.array(cls_pred == cls_gt, dtype=np.float32).mean()
    rec_recall = np.array(cls_gt * cls_pred).sum() / len(np.where((cls_gt == 1))[0])

    return total_recall, rec_recall

