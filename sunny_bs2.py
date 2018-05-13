import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC
import random
import os.path as osp
from utils import pure_profit, plot_figure

def dataloder(path=None, split_ratio=0.8, mode=None):
    """
    dataloader for sunny_bridge, split into training and validation.ratio= 8:2

    Args:
        path: the training data path.
        split_ratio: the ratio of training and validation.
        mode: the data filling format. ['average', 'sample', 'zeros']
        data_dict: {'training_data':, 'training_gt':, 'val_data':, 'val_gt', 'test_data', 'test_gt':}
    """

    path = osp.join(path, mode, 'train.data')
    data = np.load(path)

    train_input = data['train_input']
    train_target = data['train_target']

    # change the mode to synthesis classification
    # one_index = np.array(train_target[:,1] > 30, dtype=np.int16)
    # train_target[:, 0] = (train_target[:, 0] * one_index)

    test_input = data['val_input']
    test_target = data['val_target']
    length = train_input.shape[0]

    training_data = []
    training_gt = []
    val_data = []
    val_gt = []

    data_dict = {}
    for i in range(length):

        seed = np.random.random()
        if seed < split_ratio:
            training_data.append(train_input[i])
            training_gt.append(train_target[i])
        else:
            val_data.append(train_input[i])
            val_gt.append(train_target[i])

    data_dict['training_data'] = np.array(training_data, dtype=np.float64)
    data_dict['training_gt'] = np.array(training_gt, dtype=np.float64)
    data_dict['val_data'] = np.array(val_data, dtype=np.float64)
    data_dict['val_gt'] = np.array(val_gt, dtype=np.float64)
    data_dict['test_data'] = np.array(test_input, dtype=np.float64)
    data_dict['test_gt'] = np.array(test_target, dtype=np.float64)
    return data_dict


def reg_profit(data_dict=None):
    """
    This baseline can estimate the customer whether responded.
    And the ground-truth is the (responded_target \cap (profit_target>30))

    logistic regression
    data_dict:
    """
    training_data = data_dict['training_data']
    training_gt = data_dict['training_gt'][:, 1]
    training_data = training_data[np.where((training_gt!=0))[0]]
    training_gt = training_gt[np.where((training_gt!=0))[0]]

    val_data = data_dict['val_data']
    val_gt = data_dict['val_gt'][:, 1]
    val_data = val_data[np.where((val_gt != 0))[0]]
    val_gt = val_gt[np.where((val_gt != 0))[0]]

    test_data = data_dict['test_data']
    test_gt =  data_dict['test_gt'][:, 1]

    score_train_list = []
    score_val_list = []

    for c in np.linspace(1e-5, 10, 100000):
        regression_pro = Ridge(alpha=c, max_iter=10000000, tol=1e-8)
        # regression_pro = Lasso(alpha=c, max_iter=10000000, tol=1e-8)
        regression_pro.fit(training_data, training_gt)
        score_train = regression_pro .predict(training_data)
        mse_train = np.mean((score_train-training_gt)**2)
        score_val = regression_pro .predict(val_data)
        mse_val = np.mean((score_val-val_gt)**2)


        score_train_list.append(mse_train)
        score_val_list.append(mse_val)
        print('C=%.3f' % c, 'The train MSE: ', mse_train, 'The val MSE: ', mse_val)

    plot_figure(train_data=score_train_list, val_data=score_val_list, start=1e-5, stop=10, num_point=100000,
                xlabels='regularization', ylabels='mse', legends=['train', 'val'],
                save_path='./figure/profit_sample_bs2_ridge_mse.png')


def cls_response(data_dict):
    """
    This baseline can estimate the customer whether responded.
    And the ground-truth is the (responded_target \cap (profit_target>30))
    svm
    """

    training_data = data_dict['training_data']
    training_gt = data_dict['training_gt']
    val_data = data_dict['val_data']
    val_gt = data_dict['val_gt']
    test_data = data_dict['test_data']
    test_gt = data_dict['test_gt']

    score_train_list = []
    profit_train_list = []
    score_val_list = []
    profit_val_list = []
    print('train_max_profit:', np.maximum(training_gt[:, 1] - 30, 0).sum())
    print('val_max_profit:', np.maximum(val_gt[:, 1] - 30, 0).sum())
    for c in np.linspace(1e-5, 1, 100):

        svm = SVC(C=c, tol=0.0000001, max_iter=1000000, class_weight='balanced', kernel='poly')
        svm.fit(training_data, training_gt[:, 0])
        score_train = svm.score(training_data, training_gt[:, 0])
        profit_train = pure_profit(svm.predict(training_data), profit_gt=training_gt[:, 1], cls_gt=None)
        score_val = svm.score(val_data, val_gt[:, 0])
        profit_val = pure_profit(svm.predict(val_data), profit_gt=val_gt[:, 1], cls_gt=None)

        print('C=%.3f' % c, 'Score_train: %.3f, Score_val:%.3f, Profit_train: %.3f, Profit_val:%.3f'
              % (score_train, score_val, profit_train, profit_val))
    plot_figure(train_data=score_train_list, val_data=score_val_list, start=1e-5, stop=1, num_point=100,
                xlabels='regularization', ylabels='loss', legends=['train', 'val'], save_path='./figure/acc_average_bs2_svm.png')
    plot_figure(train_data=profit_train_list, val_data=profit_val_list, start=1e-5, stop=1, num_point=100,
                xlabels='regularization', ylabels='profit', legends=['train', 'val'],
                save_path='./figure/profit_average_bs2_svm.png')


def baseline2(best_cls_alpha, best_reg_alpha, data_dict):


    training_data = data_dict['training_data']
    training_gt = data_dict['training_gt'][:, 1]
    training_data_profit = training_data[np.where((training_gt != 0))[0]]
    training_gt_profit = training_gt[np.where((training_gt != 0))[0]]

    val_data = data_dict['val_data']
    val_gt = data_dict['val_gt'][:, 1]
    val_data_profit = val_data[np.where((val_gt != 0))[0]]
    val_gt_profit = val_gt[np.where((val_gt != 0))[0]]

    test_data = data_dict['test_data']
    test_gt = data_dict['test_gt'][:, 1]



def main():
    data_dict = dataloder(path='./data', split_ratio=0.8, mode='sample')
    reg_profit(data_dict=data_dict)


if __name__ == '__main__':
    np.random.seed(19)
    random.seed(19)
    main()
