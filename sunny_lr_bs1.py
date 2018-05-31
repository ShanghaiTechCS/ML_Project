# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 下午5:03
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import numpy as np
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from imblearn.over_sampling import SMOTE, ADASYN


def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, use_balance=False):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = label.float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    if use_balance:
        final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg
    else:
        final_loss = 0.5 * loss_pos + 0.5 * loss_neg

    if num_labels_pos == 0:
        final_loss = loss_neg
    if num_labels_neg == 0:
        final_loss = loss_pos

    if size_average:
        final_loss /= int(np.prod(label.size()))
    elif batch_average:
        final_loss /= int(label.size()[0])

    return final_loss


def load_data(data_path, use_tensor=False, use_cuda=False):
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

    train_input, train_target = data_package['train_input'], data_package['train_target']
    val_input, val_target = data_package['val_input'], data_package['val_target']

    train_responded, train_profit = train_target[:, 0], train_target[:, 1]
    print(train_profit)

    train_recommend = (train_responded == 1) * (train_profit > 30)

    val_responded, val_profit = val_target[:, 0], val_target[:, 1]
    val_recommend = (val_responded == 1) * (val_profit > 30)

    train_recommend = train_recommend.reshape(-1, 1)
    val_recommend = val_recommend.reshape(-1, 1)

    ret_var = (train_input, train_target, train_recommend, val_input, val_target, val_recommend)
    if use_tensor:
        if use_cuda:
            ret_var = (torch.from_numpy(val.astype(np.float32)).cuda() for val in ret_var)
        else:
            ret_var = (torch.from_numpy(val.astype(np.float32)) for val in ret_var)

    return ret_var


def compute_profit(pred_recommend, gt):
    """
    :param pred_recommend: [N, 1] value in {0, 1}
    :param gt: [N, 2]
    :return profit: float
    """

    profit = ((gt[:, 1:2] - 30) * pred_recommend).sum()
    return profit.item()


def compute_accuracy(pred_recommend, gt_recommend):
    """
    :param pred_recommend: (N, 1)
    :param gt_recommend: (N, 1)
    :return accuracy float
    """

    sample_num = gt_recommend.size(0)
    gt_recommend_num = gt_recommend.sum().item()

    total_precision = (pred_recommend == gt_recommend).sum().item() / sample_num
    recommend_recall = (pred_recommend * gt_recommend).sum().item() / gt_recommend_num

    return total_precision, recommend_recall


def compute_result(model, criterion, input_data, target_data, recommend_data, print_details=True):
    with torch.no_grad():
        pred_recommend = model(input_data)
        loss = criterion(pred_recommend, recommend_data).item()

        pred_recommend = F.sigmoid(pred_recommend)
        pred_recommend = torch.ge(pred_recommend, 0.5).float()
        profit = compute_profit(pred_recommend, target_data)
        total_precision, recommend_recall = compute_accuracy(pred_recommend, recommend_data)

        if print_details:
            print('loss:', loss)
            print('total_precision: %s, recommend_recall: %s' % (total_precision, recommend_recall))
            print('profit: ', profit)

        return loss, profit, total_precision, recommend_recall


class MLR4(nn.Module):
    def __init__(self, feature_num):
        super(MLR4, self).__init__()
        self.feature_num = feature_num

        self.fc1 = nn.Linear(feature_num, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class MLR2(nn.Module):
    def __init__(self, feature_num):
        super(MLR2, self).__init__()
        self.feature_num = feature_num

        self.fc1 = nn.Linear(feature_num, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class LR(nn.Module):
    def __init__(self, feature_num):
        super(LR, self).__init__()
        self.feature_num = feature_num

        self.fc1 = nn.Linear(feature_num, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


def logistic_regression(data_type, balance_data=False, model_name='LR'):
    """
    :param data_type: 'zero', 'average', 'sample'
    :param balance_data: bool
    :param model_name: 'MLR4', 'MLR2', 'LR'
    :return:
    """
    # load data
    train_input, train_target, train_recommend, val_input, val_target, val_recommend = load_data('new_data/train.data', use_tensor=True,
                                                                                                 use_cuda=True)

    balanced_train_input, balanced_train_recommend = SMOTE().fit_sample(train_input.cpu().numpy(), train_recommend.cpu().numpy().reshape(-1))
    balanced_train_recommend = balanced_train_recommend.reshape(-1, 1)

    balanced_train_input = torch.from_numpy(balanced_train_input.astype(np.float32)).cuda()
    balanced_train_recommend = torch.from_numpy(balanced_train_recommend.astype(np.float32)).cuda()

    print(balanced_train_input.shape)
    print(balanced_train_recommend.shape)
    total_train_num, feature_num = train_input.size()

    if model_name == 'MLR4':
        model = MLR4(feature_num).cuda()
    elif model_name == 'MLR2':
        model = MLR2(feature_num).cuda()
    elif model_name == 'LR':
        model = LR(feature_num).cuda()
    else:
        raise Exception('error model name!!!')

    criterion = class_balanced_cross_entropy_loss
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=2e-3)

    # Train the model
    num_epochs = 10000
    balanced_train_input = balanced_train_input.requires_grad_()
    balanced_train_recommend = balanced_train_recommend.requires_grad_()

    if balance_data:
        batch_num = int(train_recommend.sum().item())
        pos_idx_list = np.argwhere(train_recommend.detach().cpu().numpy()[:, 0] == 1).reshape(-1).tolist()
        neg_idx_list = list(set(list(range(total_train_num))) - set(pos_idx_list))

        # Start Train
        for epoch in range(1, num_epochs + 1):
            neg_random_idx = random.sample(neg_idx_list, batch_num)
            train_idx = pos_idx_list + neg_random_idx

            batch_train_input = train_input[train_idx, :]
            batch_train_recommend = train_recommend[train_idx, :]

            batch_train_pred_recommend = model(batch_train_input)
            loss = criterion(batch_train_pred_recommend, batch_train_recommend)
            if epoch % 100 == 0:
                print('[%d/%d]: loss: %.9f' % (epoch, num_epochs, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        # Start Train
        for epoch in range(1, num_epochs + 1):

            train_pred_recommend = model(balanced_train_input)
            loss = criterion(train_pred_recommend, balanced_train_recommend)
            if epoch % 100 == 0:
                print('[%d/%d]: loss: %.9f' % (epoch, num_epochs, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('Train:')
    compute_result(model, criterion, train_input, train_target, train_recommend)
    print('Val:')
    compute_result(model, criterion, val_input, val_target, val_recommend)


def sklearn_lr():
    # load data
    # data_path = 'new_data/train.data'
    data_path = 'data/sample/train.data'
    train_input, train_target, train_recommend, val_input, val_target, val_recommend = load_data(data_path, use_tensor=False,
                                                                                                 use_cuda=False)

    balanced_train_input, balanced_train_recommend = ADASYN().fit_sample(train_input, train_recommend.reshape(-1))
    # balanced_train_input, balanced_train_recommend = train_input, train_recommend.reshape(-1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn import svm

    clf_l1_LR = LogisticRegression(C=10, penalty='l2', tol=1e-8, max_iter=10000)
    # clf_l1_LR = svm.SVC(C=1, tol=1e-8, max_iter=1000000, class_weight='balanced', kernel='poly')

    clf_l1_LR.fit(balanced_train_input, balanced_train_recommend)
    print(classification_report(balanced_train_recommend, clf_l1_LR.predict(balanced_train_input)))

    pred = clf_l1_LR.predict(val_input)

    print(compute_profit(pred.reshape(-1, 1), val_target))

    print(classification_report(val_recommend.reshape(-1), pred))


def main():
    # logistic_regression(data_type='sample', balance_data=False, model_name='LR')
    sklearn_lr()


if __name__ == '__main__':
    main()
