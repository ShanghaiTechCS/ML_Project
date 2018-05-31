import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk



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

    predict_true = np.where((val_cls_pred==1))
    profit = np.sum(val_gt_profit[predict_true])

    return profit
