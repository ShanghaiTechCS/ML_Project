import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import pdb
import cv2
import random
import os.path as osp


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
    training_gt = []
    val_data = []
    val_gt = []

    data_dict = {}
    for i in range(length):

        seed = np.random.random()
        if seed < split_ratio:
            training_data.append(responded_input[i])
            training_gt.append(responded_synthesis_target[i])
        else:
            val_data.append(responded_input[i])
            val_gt.append(responded_synthesis_target[i])

    # import pdb
    # pdb.set_trace()
    data_dict['training_data'] = np.array(training_data, dtype=np.float64)
    data_dict['training_gt'] = np.array(training_gt, dtype=np.float64)
    data_dict['val_data'] = np.array(val_data, dtype=np.float64)
    data_dict['val_gt'] = np.array(val_gt, dtype=np.float64)

    return data_dict


def baseline1(data_dict=None):
    """
    This baseline can estimate the customer whether responded.
    And the groundtruth is the (responded_target \cap (profit_target>30))

    logistic regression
    data_dict: the data format

    """

    training_data = data_dict['training_data']
    training_gt = data_dict['training_gt']
    val_data = data_dict['val_data']
    val_gt = data_dict['val_gt']
    score_train_list = []
    score_val_list = []

    for c in np.linspace(1e-5, 100, 1000):
        logit_reg = LogisticRegression(verbose=False, class_weight='balanced', max_iter=10000, penalty='l2',
                                       tol=0.0000001,
                                       warm_start=True, n_jobs=5)
        logit_reg.fit(training_data, training_gt)
        score_train = logit_reg.score(training_data, training_gt)
        score_val = logit_reg.score(val_data, val_gt)

        score_train_list.append(score_train)
        score_val_list.append(score_val)
        print('C=%.3f' % c, 'The train mean accuracy for logistic regression: ', score_train)
        print('C=%.3f' % c, 'The val mean accuracy for logistic regression: ', score_val)

    x = np.linspace(1e-5, 100, 1000)
    plt.figure()
    plt.plot(x, np.array(score_train_list))
    plt.plot(x, np.array(score_val_list))
    plt.xlabel('regularization strength')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'])
    plt.savefig('./figure/baseline1_logR.png')


def baseline2(data_dict):
    """
    This baseline can estimate the customer whether responded.
    And the groundtruth is the (responded_target \cap (profit_target>30))
    svm
    """

    training_data = data_dict['training_data']
    training_gt = data_dict['training_gt']
    val_data = data_dict['val_data']
    val_gt = data_dict['val_gt']
    score_train_list = []
    score_val_list = []

    for c in np.linspace(1, 100, 100):

        svm = SVC(C=c, tol=0.0000001, max_iter=1000000, class_weight='balanced', kernel='poly')
        svm.fit(training_data, training_gt)
        score_train = svm.score(training_data, training_gt)
        score_val = svm.score(val_data, val_gt)

        score_train_list.append(score_train)
        score_val_list.append(score_val)
        print('C=%.3f' % c, 'The train mean accuracy for SVM: ', score_train)
        print('C=%.3f' % c, 'The val mean accuracy for SVM: ', score_val)

    x = np.linspace(1, 100, 100)
    plt.figure()
    plt.plot(x, np.array(score_train_list))
    plt.plot(x, np.array(score_val_list))
    plt.xlabel('regularization strength')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'])
    # plt.show()
    plt.savefig('./figure/baseline2_svm_poly.png')


def plot_figure():
    pass


def main():
    data_dict = dataloder(path='./data/zero', split_ratio=0.8)
    # baseline1(data_dict=data_dict)
    baseline2(data_dict=data_dict)


if __name__ == '__main__':
    np.random.seed(19)
    random.seed(19)
    main()
