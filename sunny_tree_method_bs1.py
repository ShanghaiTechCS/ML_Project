# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 下午2:01
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn


import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


def load_data(data_path):
    """
    :param data_path:
    :return: data_package = {
            feature_standard_weight_list: 55
            train_input: (7323, 55)
            train_target: (7323, 2)
            val_input: (814, 55)
            val_target: (814, 2)
                    }
    """
    with open(data_path, 'rb') as f:
        data_package = pickle.load(f)
    return data_package


def compute_profit(pred_recommend, gt):
    """
    :param pred_recommend: [N,] value in {0, 1}
    :param gt: [N, 2]
    :return profit: float
    """

    profit = ((gt[:, 1] - 30) * pred_recommend).sum()
    return profit


def compute_accuracy(pred_recommend, gt):
    """
    :param pred_recommend: (N, )
    :param gt: (N, 2)
    :return accuracy float
    """

    gt_responded, gt_profit = gt[:, 0], gt[:, 1]
    gt_recommend = (gt_responded == 1) * (gt_profit > 30)
    sample_num = gt_recommend.shape[0]
    gt_recommend_num = gt_recommend.sum()

    total_precision = (pred_recommend == gt_recommend).sum() / sample_num
    recommend_recall = (pred_recommend * gt_recommend).sum() / gt_recommend_num

    return total_precision, recommend_recall


def normal_DT(data_type):
    def test_param(min_samples_split, max_depth):
        clf = DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_split=min_samples_split, max_depth=max_depth)
        clf.fit(train_input, train_recommend)

        pred_train_recommend = clf.predict(train_input)
        pred_val_recommend = clf.predict(val_input)

        train_total_precision, train_recommend_recall = compute_accuracy(pred_train_recommend, train_target)
        train_profit = compute_profit(pred_train_recommend, train_target)

        val_total_precision, val_recommend_recall = compute_accuracy(pred_val_recommend, val_target)
        val_profit = compute_profit(pred_val_recommend, val_target)

        print('train_total_precision: %s, train_recommend_recall: %s' % (train_total_precision, train_recommend_recall))
        print('train_profit: ', train_profit)
        print('-' * 30)

        print('val_total_precision: %s, val_recommend_recall: %s' % (val_total_precision, val_recommend_recall))
        print('val_profit: ', val_profit)

        result = {'train_TP': train_total_precision, 'train_RR': train_recommend_recall, 'train_profit': train_profit,
                  'val_TP': val_total_precision, 'val_RR': val_recommend_recall, 'val_profit': val_profit}

        return result

    data_package = load_data('data/%s/train.data' % data_type)
    train_input, train_target = data_package['train_input'], data_package['train_target']
    val_input, val_target = data_package['val_input'], data_package['val_target']

    train_responded, train_profit = train_target[:, 0], train_target[:, 1]
    train_recommend = (train_responded == 1) * (train_profit > 30)

    val_responded, val_profit = val_target[:, 0], val_target[:, 1]
    val_recommend = (val_responded == 1) * (val_profit > 30)

    val_profit_list = []
    for min_samples_split in range(5, 100, 5):
        for max_depth in range(1, 30):
            result = test_param(min_samples_split, max_depth)
            val_profit_list.append((result, min_samples_split, max_depth))

    val_profit_list = sorted(val_profit_list, key=lambda x: x[0]['val_profit'], reverse=True)
    print(val_profit_list[0])


def main():
    normal_DT('sample')


if __name__ == '__main__':
    main()
