import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
import pandas as pd
import pdb
import cv2
import random
import os.path as osp
from utils import plot_figure


def dataloder(path=None, split_ratio=0.8):
    """
    dataloader for sunny_bridge, split into training and validation.ratio= 8:2

    Args:
        path: the training data path.
        split_ratio: the ratio of training and validation.
        data_dict: {'training_data':, 'training_gt':, 'val_data':, 'val_gt':}
    """

    feat_attr = ['custAge', 'profession', 'marital', 'schooling', 'default', 'housing', 'loan', 'contact', 'month',
                 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                 'cons.conf.idx', 'euribor3m', 'nr.employed', 'pastEmail']

    path = osp.join(path, 'train.data')
    data = np.load(path)

    responded_input = data['responded_input']
    responded_target = data['responded_target']
    profit_input = data['profit_input']
    profit_target = data['profit_target']
    length = responded_input.shape[0]

    responded_synthesis = responded_target[7310:].squeeze().astype(np.int8) & (profit_target > 30).squeeze().astype(
        np.int8)
    print('num_postive:', np.sum(responded_synthesis))

    responded_synthesis_target = np.concatenate((responded_target[:7310].squeeze(), responded_synthesis))
    training_data = []
    training_gt_sys = []
    training_gt_res = []
    val_data = []
    training_data_profit = []
    val_data_profit = []
    training_gt_profit = []
    val_gt_profit = []
    val_gt_sys = []
    val_gt_res = []

    data_dict = {}
    for i in range(length):

        seed = np.random.random()
        if seed < split_ratio:
            training_data.append(responded_input[i])
            training_gt_sys.append(responded_synthesis_target[i])
            training_gt_res.append(responded_target[i])
            if responded_target[i] == 1:
                training_data_profit.append(responded_input[i])
                training_gt_profit.append(profit_target[i-7310])
        else:
            val_data.append(responded_input[i])
            val_gt_sys.append(responded_synthesis_target[i])
            val_gt_res.append(responded_target[i])

            if responded_target[i] == 1:
                val_data_profit.append(responded_input[i])
                val_gt_profit.append(profit_target[i-7310])

    ## for cls
    data_dict['training_data'] = np.array(training_data, dtype=np.float64)
    data_dict['training_gt_res'] = np.array(training_gt_res, dtype=np.float64)
    data_dict['training_gt_sys'] = np.array(training_gt_sys, dtype=np.float64)
    data_dict['val_data'] = np.array(val_data, dtype=np.float64)
    data_dict['val_gt_sys'] = np.array(val_gt_sys, dtype=np.float64)
    data_dict['val_gt_res'] = np.array(val_gt_res, dtype=np.float64)

    ## data for regression
    data_dict['training_data_profit'] = np.array(training_data, dtype=np.float64)
    data_dict['val_data_profit'] = np.array(training_data, dtype=np.float64)
    data_dict['training_gt_profit'] = np.array(training_data, dtype=np.float64)
    data_dict['val_gt_profit'] = np.array(training_data, dtype=np.float64)


    return data_dict


def reg_profit(data_dict=None):
    """
    This baseline can estimate the customer whether responded.
    And the ground-truth is the (responded_target \cap (profit_target>30))

    logistic regression
    data_dict:

    """

    training_data = data_dict['training_data_profit']
    training_gt = data_dict['training_gt_profit']
    val_data = data_dict['val_data_profit']
    val_gt = data_dict['val_gt_profit']
    score_train_list = []
    score_val_list = []

    for c in np.linspace(1e-6, 1000, 100000):
        regression_pro = Ridge(alpha=c, max_iter=10000000000, tol=1e-9, solver='auto')
        regression_pro.fit(training_data, training_gt)
        score_train = regression_pro .score(training_data, training_gt)
        score_val = regression_pro .score(val_data, val_gt)

        score_train_list.append(score_train)
        score_val_list.append(score_val)
        print('C=%.3f' % c, 'The train MSE: ', score_train)
        print('C=%.3f' % c, 'The val MSE: ', score_val)

    x = np.linspace(1e-6, 1000, 100000)
    plot_figure(training=score_train_list, validation=score_val_list, stop=1000, start=1e-6, num_point=100000, ylabel='mse',
                legend=['training', 'validation'], name='./figure/baseline2_reg_profit.png')


def cls_responsed(data_dict):
    """
    This baseline can estimate the customer whether responded.
    And the groundtruth is the (responded_target \cap (profit_target>30))

    svm

    """

    training_data = data_dict['training_data']
    training_gt = data_dict['training_gt_res']
    val_data = data_dict['val_data']
    val_gt = data_dict['val_gt_res']
    score_train_list = []
    score_val_list = []

    acc_true_list = []
    acc_false_list = []

    for c in np.linspace(1e-5, 50, 100):
        svm = SVC(C=c, tol=0.0000001, max_iter=1000000, class_weight='balanced', kernel='poly')

        svm.fit(training_data, training_gt)
        score_train = svm.score(training_data, training_gt)
        score_val = svm.score(val_data, val_gt)
        predict = svm.predict(val_data)

        index_true = np.where((val_gt == 1))[0]
        index_false = np.where((val_gt == 0))[0]
        acc_true = np.mean(predict[index_true])
        acc_false = np.mean(1 - predict[index_false])
        acc_true_list.append(acc_true)
        acc_false_list.append(acc_false)

        score_train_list.append(score_train)
        score_val_list.append(score_val)
        print('C=%.3f' % c, 'The train mean accuracy for SVM: ', score_train)
        print('C=%.3f' % c, 'The val mean accuracy for SVM: ', score_val)
        print('C=%.3f' % c, 'The val true accuracy for SVM: ', acc_true)
        print('C=%.3f' % c, 'The val true false accuracy for SVM: ', acc_false)
        print('------------------------------------------------')
        print('')

    score_train_list = np.array(score_train_list)
    score_val_list = np.array(score_val_list)
    acc_true_list = np.array(acc_true_list)
    acc_false_list = np.array(acc_false_list)

    plot_figure(score_train_list, score_val_list, start=1e-5, stop=50, num_point=100,
                name='./figure/baseline2_svm_poly_2_target.png', ylabel='acc', legend=['training', 'validation'])

    plot_figure(acc_false_list, acc_true_list, start=1e-5, stop=50, num_point=100,
                name='./figure/baseline2_svm_poly_2_recall_target.png', ylabel='acc', legend=['false', 'true'])


def main():

    data_dict = dataloder(path='./data/zero', split_ratio=0.8)
    # cls_responsed(data_dict=data_dict)
    reg_profit(data_dict=data_dict)

if __name__ == '__main__':
    np.random.seed(19)
    random.seed(19)
    main()
