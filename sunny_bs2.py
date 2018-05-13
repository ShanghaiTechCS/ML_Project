import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC
import random
from utils import recall_cls
import os.path as osp
from utils import pure_profit, plot_figure, classification_fusion


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
    training_data = training_data[np.where((training_gt != 0))[0]]
    training_gt = training_gt[np.where((training_gt != 0))[0]]

    val_data = data_dict['val_data']
    val_gt = data_dict['val_gt'][:, 1]
    val_data = val_data[np.where((val_gt != 0))[0]]
    val_gt = val_gt[np.where((val_gt != 0))[0]]

    test_data = data_dict['test_data']
    test_gt = data_dict['test_gt'][:, 1]

    score_train_list = []
    score_val_list = []

    # for c in np.linspace(1e-5, 10, 100000):
    for c in [5]:
        # regression_pro = Ridge(alpha=c, max_iter=10000000, tol=1e-8)
        # regression_pro = Lasso(alpha=c, max_iter=10000000, tol=1e-8)
        regression_pro = MLPRegressor(hidden_layer_sizes=(64, 128, 64), activation='relu', solver='adam', alpha=0.00001,
                                      batch_size='auto', learning_rate='adaptive', max_iter=1000000, shuffle=True,
                                      tol=1e-48, momentum=0.99, verbose=True, warm_start=True)
        regression_pro.fit(training_data, training_gt)
        score_train = regression_pro.predict(training_data)
        mse_train = np.mean((score_train - training_gt) ** 2)
        score_val = regression_pro.predict(val_data)
        mse_val = np.mean((score_val - val_gt) ** 2)
        import pdb
        pdb.set_trace()
        score_train_list.append(mse_train)
        score_val_list.append(mse_val)
        print('C=%.3f' % c, 'The train MSE: ', mse_train, 'The val MSE: ', mse_val)

    plot_figure(train_data=score_train_list, val_data=score_val_list, start=1e-5, stop=10, num_point=100000,
                xlabels='regularization', ylabels='mse', legends=['train', 'val'],
                save_path='./figure/profit_sample_bs2_mlp_mse.png')


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
                xlabels='regularization', ylabels='loss', legends=['train', 'val'],
                save_path='./figure/acc_average_bs2_svm.png')
    plot_figure(train_data=profit_train_list, val_data=profit_val_list, start=1e-5, stop=1, num_point=100,
                xlabels='regularization', ylabels='profit', legends=['train', 'val'],
                save_path='./figure/profit_average_bs2_svm.png')


def cls_response_mlp(data_dict):
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


    print('train_max_profit:', np.maximum(training_gt[:, 1] - 30, 0).sum())
    print('val_max_profit:', np.maximum(val_gt[:, 1] - 30, 0).sum())

    mlpcls = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.001,
                           batch_size='auto', verbose=True, learning_rate='adaptive', warm_start=True, momentum=0.99,
                           max_iter=10000, shuffle=True, learning_rate_init=0.001, tol=1e-8)
    mlpcls.fit(training_data, training_gt[:, 0])

    cls_pred_train = mlpcls.predict(training_data)
    cls_pred_val = mlpcls.predict(val_data)

    total_acc_train, rec_acc_train = recall_cls(cls_pred_train, training_gt[:, 0])
    profit_train = pure_profit(cls_pred_train, profit_gt=training_gt[:, 1], cls_gt=None)
    total_acc_val, rec_acc_val = recall_cls(cls_pred_val, val_gt[:, 0])
    profit_val = pure_profit(cls_pred_val, profit_gt=val_gt[:, 1], cls_gt=None)

    print('total_acc_train: %.3f, rec_acc_train:%.3f, total_acc_val:%.3f, rec_acc_val:%.3f, Profit_train: %.3f, '
          'Profit_val:%.3f'
          % (total_acc_train, rec_acc_train, total_acc_val,  rec_acc_val, profit_train, profit_val))


def baseline2(best_cls_alpha, best_reg_alpha, data_dict):
    training_data = data_dict['training_data']
    training_gt = data_dict['training_gt'][:, 1]
    training_data_profit = training_data[np.where((training_gt != 0))[0]]
    training_gt_profit = training_gt[np.where((training_gt != 0))[0]]

    train_gt_cls = data_dict['training_gt'][:, 0]

    val_data = data_dict['val_data']
    val_gt = data_dict['val_gt'][:, 1]
    val_data_profit = val_data[np.where((val_gt != 0))[0]]
    val_gt_profit = val_gt[np.where((val_gt != 0))[0]]
    val_gt_cls = data_dict['val_gt'][:, 0]

    test_data = data_dict['test_data']
    test_gt = data_dict['test_gt'][:, 1]

    reg = Lasso(alpha=best_reg_alpha, max_iter=10000000, tol=1e-8)
    svm = SVC(C=best_cls_alpha, tol=0.0000001, max_iter=1000000, class_weight='balanced', kernel='poly')

    # training mode
    reg.fit(training_data_profit, training_gt_profit)
    svm.fit(training_data, train_gt_cls)

    ## inference mode
    cls_result_train = svm.predict(training_data)
    cls_result_val = svm.predict(val_data)

    reg_result_train = reg.predict(training_data)
    reg_result_val = reg.predict(val_data)

    ## visualize the resutls
    print('best_cls_alpha:', best_cls_alpha, 'best_reg_alpha:', best_reg_alpha)

    acc = svm.score(val_data, val_gt_cls)
    mse = np.mean((reg.predict(val_data_profit) - val_gt_profit) ** 2)
    import pdb
    pdb.set_trace()

    print('acc:', acc, 'MSE:', mse)

    profit_train = []
    profit_val = []

    for thres_high in range(30, 200, 5):
        cls_result_fusion_train = classification_fusion(cls_pred=cls_result_train, reg_pred=reg_result_train,
                                                        thres_high=thres_high, thres_low=-1000)
        reg_profit_train = pure_profit(cls_pred=cls_result_fusion_train, cls_gt=None, profit_gt=training_gt)

        cls_result_fusion_val = classification_fusion(cls_pred=cls_result_val, reg_pred=reg_result_val,
                                                      thres_high=thres_high, thres_low=-1000)
        reg_profit_val = pure_profit(cls_pred=cls_result_fusion_val, cls_gt=None, profit_gt=val_gt)

        profit_train.append(reg_profit_train)
        profit_val.append(reg_profit_val)

    profit_val = np.array(profit_val, dtype=np.float32)
    profit_train = np.array(profit_train, dtype=np.float32)

    plot_figure(train_data=profit_train, val_data=profit_val, legends=['train', 'val'], xlabels='threshold', start=5,
                stop=195, num_point=34, save_path='./figure/combine_profit_bs2.png', ylabels='profit')


def main():
    data_dict = dataloder(path='./data', split_ratio=0.8, mode='sample')
    # reg_profit(data_dict=data_dict)
    # baseline2(best_cls_alpha=0.364, best_reg_alpha=1.8, data_dict=data_dict)
    cls_response_mlp(data_dict=data_dict)


if __name__ == '__main__':
    np.random.seed(19)
    random.seed(19)
    main()
