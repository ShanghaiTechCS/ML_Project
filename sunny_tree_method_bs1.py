# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 下午2:01
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn


import numpy as np
import matplotlib.pyplot as plt
import pickle
import lightgbm as lgb

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


def lightGBM(data_type):
    def test_param(num_leaves, max_depth, min_data_in_leaf):
        # specify your configurations as a dict
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            # 'max_depth': 1,
            'objective': 'binary',
            'metric': {'binary', 'auc'},
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'num_iterations': 10000,
            'min_data_in_leaf': min_data_in_leaf,
            'is_unbalance': True
        }

        print('Start training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=200,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=5)

        print('Save model...')
        # save model to file
        gbm.save_model('model.txt')

        print('Start predicting...')
        # predict
        pred_train_recommend = gbm.predict(train_input, num_iteration=gbm.best_iteration) > 0.5
        pred_val_recommend = gbm.predict(val_input, num_iteration=gbm.best_iteration) > 0.5

        # eval
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

    # load or create your dataset
    data_package = load_data('data/%s/train.data' % data_type)
    train_input, train_target = data_package['train_input'], data_package['train_target']
    val_input, val_target = data_package['val_input'], data_package['val_target']

    train_responded, train_profit = train_target[:, 0], train_target[:, 1]
    train_recommend = (train_responded == 1) * (train_profit > 30)

    val_responded, val_profit = val_target[:, 0], val_target[:, 1]
    val_recommend = (val_responded == 1) * (val_profit > 30)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_input, train_recommend)
    lgb_eval = lgb.Dataset(val_input, val_recommend, reference=lgb_train)

    # find max profit
    val_profit_list = []
    for max_depth in range(2, 20, 1):
        for min_data_in_leaf in range(10, 200, 10):
            num_leaves_step = 2 ** (max_depth - 1) // 20
            num_leaves_step = 1 if num_leaves_step == 0 else num_leaves_step
            for num_leaves in range(2 ** (max_depth - 1), 2 ** max_depth, num_leaves_step):
                result = test_param(num_leaves, max_depth, min_data_in_leaf)
                val_profit_list.append((result, num_leaves, max_depth, min_data_in_leaf))

    val_profit_list = sorted(val_profit_list, key=lambda x: x[0]['val_profit'], reverse=True)
    print(val_profit_list[0])


def main():
    # normal_DT('sample')
    lightGBM('sample')


if __name__ == '__main__':
    main()
