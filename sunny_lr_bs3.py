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


def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, use_balance=True):
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
            feature_standard_weight_list: 54
            train_input: (7323, 54)
            train_target: (7323, 2)
            train_cust: (7323, 29)
            train_cust: (7323, 29)
            val_cust: (814, 29)
            val_target: (814, 2)
                    }
    """
    with open(data_path, 'rb') as f:
        data_package = pickle.load(f)

    train_input, train_cust, train_target = data_package['train_input'], data_package['train_cust'], data_package['train_target']
    val_input, val_cust, val_target = data_package['val_input'], data_package['val_cust'], data_package['val_target']

    train_cust = train_cust[:, -5:]
    val_cust = val_cust[:, -5:]

    train_responded, train_profit = train_target[:, 0:1], train_target[:, 1:2]
    train_recommend = (train_responded == 1) * (train_profit > 30)

    val_responded, val_profit = val_target[:, 0:1], val_target[:, 1:2]
    val_recommend = (val_responded == 1) * (val_profit > 30)

    train_recommend = train_recommend.reshape(-1, 1)
    val_recommend = val_recommend.reshape(-1, 1)

    ret_var = (train_input, train_cust, train_responded, train_profit, train_recommend,
               val_input, val_cust, val_responded, val_profit, val_recommend)
    if use_tensor:
        if use_cuda:
            ret_var = (torch.from_numpy(val.astype(np.float32)).cuda() for val in ret_var)
        else:
            ret_var = (torch.from_numpy(val.astype(np.float32)) for val in ret_var)

    return ret_var


def compute_profit(pred_recommend, gt_profit):
    """
    :param pred_recommend: [N, 1] value in {0, 1}
    :param gt_profit: [N, 1]
    :return profit: float
    """

    profit = ((gt_profit - 30) * pred_recommend).sum()
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


class MLR4(nn.Module):
    def __init__(self, feature_num):
        super(MLR4, self).__init__()
        self.feature_num = feature_num

        self.fc1 = nn.Linear(feature_num, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
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


class MLR3(nn.Module):
    def __init__(self, feature_num):
        super(MLR3, self).__init__()
        self.feature_num = feature_num

        self.fc1 = nn.Linear(feature_num, 16)
        self.fc2 = nn.Linear(16, 16)
        # self.fc3 = nn.Linear(16, 1)
        # self.fc4 = nn.Linear(32, 32)
        # self.fc5 = nn.Linear(16, 1)

        self.fc0p = nn.Linear(feature_num, 1)
        self.fc1p = nn.Linear(16, 1)
        self.fc2p = nn.Linear(16, 1)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(16)

    def forward(self, x):
        x1 = self.bn1(F.relu(self.fc1(x)))
        x2 = self.bn2(F.relu(self.fc2(x1)))
        # x3 = self.bn3(F.relu(self.fc3(x2))) + x2
        # x4 = self.bn4(F.relu(self.fc4(x3))) + x3

        x0p = self.fc0p(x)
        x1p = self.fc1p(x1)
        x2p = self.fc1p(x2)

        x = x0p + x1p + x2p

        return x


class LR(nn.Module):
    def __init__(self, feature_num):
        super(LR, self).__init__()
        self.feature_num = feature_num

        self.fc1 = nn.Linear(feature_num, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


class MultiTaskModel(nn.Module):
    def __init__(self, total_feature_num, cust_feature_num):
        super(MultiTaskModel, self).__init__()

        # self.fc1 = nn.Linear(feature_num, 32)
        # self.fc2 = nn.Linear(32, 32)
        # self.fc3 = nn.Linear(64, 64)
        # self.fc4 = nn.Linear(64, 32)

        self.responded_model = LR(total_feature_num)
        self.profit_model = MLR3(cust_feature_num)

    def forward(self, total_x, cust_x):
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        # x = self.fc1(x)

        responded = self.responded_model(total_x)
        profit = self.profit_model(cust_x)
        return responded, profit


def compute_result(model, input_data, cust_data, responded_data, profit_data, recommend_data, print_details=True):
    with torch.no_grad():
        pred_responded, pred_profit = model(input_data, cust_data)
        loss, responded_loss, profit_loss, final_profit_loss = multi_task_loss(pred_responded, pred_profit, responded_data, profit_data)
        loss, responded_loss, profit_loss, final_profit_loss = loss.item(), responded_loss.item(), profit_loss.item(), final_profit_loss.item()

        pred_responded = F.sigmoid(pred_responded)
        # pred_recommend = torch.ge(pred_responded, 0.4).float() * torch.ge(pred_profit, 30).float()
        # pred_recommend = torch.ge(pred_recommend, 30).float()

        pred_recommend = torch.ge(pred_responded * pred_profit, 30).float()

        profit = compute_profit(pred_recommend, profit_data)
        total_precision, recommend_recall = compute_accuracy(pred_recommend, recommend_data)

        if print_details:
            print('loss: %.9f responded_loss: %.9f, profit_loss: %.9f, final_profit_loss: %.9f' % (loss, responded_loss, profit_loss, final_profit_loss))
            print('total_precision: %s, recommend_recall: %s' % (total_precision, recommend_recall))
            print('profit: ', profit)

        return loss, profit, total_precision, recommend_recall


def multi_task_loss(pred_responded, pred_profit, gt_responded, gt_profit):
    # sample num
    sample_num = gt_responded.size(0)

    # responded loss
    responded_loss = class_balanced_cross_entropy_loss(pred_responded, gt_responded)

    # profit loss
    pos_num = (gt_responded == 1).sum().item()
    neg_num = sample_num - pos_num

    pos_pred_profit = pred_profit[gt_responded == 1]
    pos_gt_profit = gt_profit[gt_responded == 1]
    pos_profit_loss = ((pos_pred_profit - pos_gt_profit) ** 2).sum() / sample_num / 2

    neg_pred_profit = pred_profit[gt_responded == 0]
    neg_gt_profit = gt_profit[gt_responded == 0]
    neg_profit_loss = ((neg_pred_profit - neg_gt_profit) ** 2).sum() / sample_num / 2

    profit_loss = 4e-4 * (neg_num / sample_num * pos_profit_loss + pos_num / sample_num * neg_profit_loss)

    # final profit
    gt_final_profit = gt_profit[gt_profit >= 30].sum() / sample_num
    pred_final_profit = (F.sigmoid(pred_responded) * pred_profit).mean()
    final_profit_loss = 1e-3 * (gt_final_profit - pred_final_profit) ** 2 / 2

    total_loss = responded_loss + profit_loss + final_profit_loss
    return total_loss, responded_loss, profit_loss, final_profit_loss


def logistic_regression(data_type, model_name):
    """
    :param data_type: 'zero', 'average', 'sample'
    :param model_name: 'MLR4', 'MLR2', 'LR'
    :return:
    """
    # load data
    train_input, train_cust, train_responded, train_profit, train_recommend, val_input, val_cust, val_responded, val_profit, val_recommend = \
        load_data('data/%s/train.data' % data_type, use_tensor=True, use_cuda=True)
    total_train_num, total_feature_num = train_input.size()
    cust_feature_num = train_cust.size(1)

    if model_name == 'MLR4':
        base_model = MLR4
    elif model_name == 'MLR2':
        base_model = MLR2
    elif model_name == 'LR':
        base_model = LR
    else:
        raise Exception('error model name!!!')

    model = MultiTaskModel(total_feature_num, cust_feature_num).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=2e-3)

    # Train the model
    num_epochs = 10000
    train_input, train_cust, train_responded = train_input.requires_grad_(), train_cust.requires_grad_(), train_responded.requires_grad_()

    # Start Train
    for epoch in range(1, num_epochs + 1):
        train_pred_responded, train_pred_profit = model(train_input, train_cust)
        loss, responded_loss, profit_loss, final_profit_loss = multi_task_loss(train_pred_responded, train_pred_profit, train_responded, train_profit)

        if epoch % 100 == 0:
            print('[%d/%d]: loss: %.9f, responded_loss: %.9f, profit_loss: %.9f, final_profit_loss: %.9f' % (
                epoch, num_epochs, loss.item(), responded_loss.item(), profit_loss.item(), final_profit_loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train:')
    compute_result(model, train_input, train_cust, train_responded, train_profit, train_recommend)
    print('Val:')
    compute_result(model, val_input, val_cust, val_responded, val_profit, val_recommend)


def main():
    logistic_regression(data_type='sample', model_name='MLR4')


if __name__ == '__main__':
    main()
