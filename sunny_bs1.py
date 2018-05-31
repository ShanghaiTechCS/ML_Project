import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random
import os.path as osp
from utils import pure_profit, plot_figure2, confusion_matrix_compute
from imblearn.over_sampling import SMOTE, ADASYN

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

    import pickle
    path = osp.join(path, 'train.data')
    pickle_file = open(path, 'rb')
    data = pickle.load(pickle_file)

    train_input = data['train_input']
    train_target = data['train_target']
    val_input = data['val_input']
    val_target = data['val_target']


    train_profit = train_target[:, 1]
    train_cls = train_target[:, 0]

    positive_profit_index = np.where((train_profit>=30))[0]
    train_gt_cls = np.zeros_like(train_cls)
    train_gt_cls[positive_profit_index] = 1
    train_target[:, 0] = train_gt_cls

    val_profit = val_target[:, 1]
    val_cls = val_target[:, 0]
    positive_profit_index = np.where((val_profit >= 30))[0]
    val_gt_cls = np.zeros_like(val_cls)
    val_gt_cls[positive_profit_index] = 1
    val_target[:, 0] = val_gt_cls

    ## for cls
    data_dict = {}
    data_dict['training_data'] = np.array(train_input, dtype=np.float64)
    data_dict['training_gt'] = np.array(train_target, dtype=np.float64)
    data_dict['val_data'] = np.array(val_input, dtype=np.float64)
    data_dict['val_gt'] = np.array(val_target, dtype=np.float64)
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


    total_acc_train = []
    pos_acc_train = []
    neg_acc_train = []

    profit_train_list = []
    profit_val_list = []
    total_acc_val = []
    pos_acc_val = []
    neg_acc_val = []

    for c in np.linspace(1e-5, 3, 1000):
        logit_reg = LogisticRegression(verbose=False, class_weight='balanced', max_iter=10000, penalty='l2',
                                       tol=0.0000001,
                                       warm_start=True, n_jobs=1, C=c)
        training_data_resample, training_gt_resample = SMOTE().fit_sample(training_data, training_gt[:, 0])
        logit_reg.fit(training_data_resample, training_gt_resample)

        train_pred = logit_reg.predict(training_data)
        profit_train = pure_profit(train_pred, profit_gt=training_gt[:, 1], cls_gt=None)
        val_pred = logit_reg.predict(val_data)
        profit_val = pure_profit(val_pred, profit_gt=val_gt[:, 1], cls_gt=None)

        total_acc_t, pos_acc_t, neg_acc_t = confusion_matrix_compute(cls_pred=train_pred, cls_gt= training_gt[:, 0])
        total_acc_v, pos_acc_v, neg_acc_v = confusion_matrix_compute(cls_pred=val_pred, cls_gt= val_gt[:, 0])

        total_acc_train.append(total_acc_t)
        total_acc_val.append(total_acc_v)
        pos_acc_train.append(pos_acc_t)
        pos_acc_val.append(pos_acc_v)
        neg_acc_train.append(neg_acc_t)
        neg_acc_val.append(neg_acc_v)
        profit_train_list.append(profit_train)
        profit_val_list.append(profit_val)



        print('C=%.3f' % c, 'Socre_train: %.3f, Score_val:%.3f, pos_train:%3f,pos_val:%3f,neg_train:%3f,neg_val:%3f,Profit_train: %.3f, Profit_val:%.3f'
              %(total_acc_t, total_acc_v, pos_acc_t, pos_acc_v, neg_acc_t, neg_acc_v,profit_train, profit_val))


    plot_figure2(train_data=total_acc_train, val_data=total_acc_val, start=1e-5, stop=3, num_point=1000,
                xlabels='regularization', ylabels='total_acc', legends=['train', 'val'], save_path='./new_fig/acc_bs1_total_smote.png')
    plot_figure2(train_data=pos_acc_train, val_data=pos_acc_val, start=1e-5, stop=3, num_point=1000,
                xlabels='regularization',ylabels='Pos_acc', legends=['train', 'val'], save_path='./new_fig/acc_bs1_pos_smote.png')
    plot_figure2(train_data=profit_train_list, val_data=profit_val_list, start=1e-5, stop=3, num_point=1000,
                xlabels='regularization', ylabels='profit', legends=['train', 'val'], save_path='./new_fig/profit_bs1_lg_smote.png')


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

    total_acc_train = []
    pos_acc_train = []
    neg_acc_train = []

    profit_train_list = []
    profit_val_list = []
    total_acc_val = []
    pos_acc_val = []
    neg_acc_val = []

    print('train_max_profit:', np.maximum(training_gt[:, 1] - 30, 0).sum())
    print('val_max_profit:', np.maximum(val_gt[:, 1] - 30, 0).sum())
    for c in np.linspace(1e-5, 1, 100):

        svm = SVC(C=c, tol=0.0000001, max_iter=1000000, class_weight='balanced', kernel='poly')
        training_data_resample, training_gt_resample = SMOTE().fit_sample(training_data, training_gt[:, 0])
        svm.fit(training_data_resample, training_gt_resample)

        train_pred = svm.predict(training_data)
        profit_train = pure_profit(train_pred, profit_gt=training_gt[:, 1], cls_gt=None)
        val_pred = svm.predict(val_data)
        profit_val = pure_profit(val_pred, profit_gt=val_gt[:, 1], cls_gt=None)

        total_acc_t, pos_acc_t, neg_acc_t = confusion_matrix_compute(cls_pred=train_pred, cls_gt=training_gt[:, 0])
        total_acc_v, pos_acc_v, neg_acc_v = confusion_matrix_compute(cls_pred=val_pred, cls_gt=val_gt[:, 0])

        total_acc_train.append(total_acc_t)
        total_acc_val.append(total_acc_v)
        pos_acc_train.append(pos_acc_t)
        pos_acc_val.append(pos_acc_v)
        neg_acc_train.append(neg_acc_t)
        neg_acc_val.append(neg_acc_v)
        profit_train_list.append(profit_train)
        profit_val_list.append(profit_val)

        print('C=%.3f' % c,
              'Socre_train: %.3f, Score_val:%.3f, pos_train:%3f,pos_val:%3f,neg_train:%3f,neg_val:%3f,Profit_train: %.3f, Profit_val:%.3f'
              % (total_acc_t, total_acc_v, pos_acc_t, pos_acc_v, neg_acc_t, neg_acc_v, profit_train, profit_val))

    plot_figure2(train_data=total_acc_train, val_data=total_acc_val, start=1e-5, stop=1, num_point=100,
                 xlabels='regularization', ylabels='total_acc', legends=['train', 'val'],
                 save_path='./new_fig/acc_bs1_total_svm_poly_smote.png')
    plot_figure2(train_data=pos_acc_train, val_data=pos_acc_val, start=1e-5, stop=1, num_point=100,
                 xlabels='regularization', ylabels='Pos_acc', legends=['train', 'val'],
                 save_path='./new_fig/acc_bs1_pos_svm_poly_smote.png')
    plot_figure2(train_data=profit_train_list, val_data=profit_val_list, start=1e-5, stop=1, num_point=100,
                 xlabels='regularization', ylabels='profit', legends=['train', 'val'],
                 save_path='./new_fig/profit_bs1_svm_poly_smote.png')


def main():
    data_dict = dataloder(path='./new_data', split_ratio=0.8)
    # baseline1(data_dict=data_dict)
    baseline2(data_dict=data_dict)



if __name__ == '__main__':
    np.random.seed(19)
    random.seed(19)
    main()
