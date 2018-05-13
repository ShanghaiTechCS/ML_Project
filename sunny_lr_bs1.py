# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 下午5:03
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MLR(nn.Module):
    def __init__(self, feature_num):
        super(MLR, self).__init__()
        self.feature_num = feature_num

        self.fc1 = nn.Linear(feature_num, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = self.fc4(x3)

        return x4


def logistic_regression(data_type):
    # load data
    train_input, train_target, train_recommend, val_input, val_target, val_recommend = load_data('data/%s/train.data' % data_type, use_tensor=True,
                                                                                                 use_cuda=True)
    feature_num = train_input.size(1)

    model = MLR(feature_num).cuda()

    criterion = class_balanced_cross_entropy_loss
    # optimizer = torch.optim.SGD(model.parameters(), lr=2e-2, weight_decay=1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=2e-3)

    # Train the model
    num_epochs = 3000
    train_input, train_recommend = train_input.requires_grad_(), train_recommend.requires_grad_()
    for epoch in range(1, num_epochs + 1):
        train_pred_recommend = model(train_input)
        loss = criterion(train_pred_recommend, train_recommend)
        if epoch % 100 == 0:
            print('[%d/%d]: loss: %.9f' % (epoch, num_epochs, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train:')
    compute_result(model, criterion, train_input, train_target, train_recommend)
    print('Val:')
    compute_result(model, criterion, val_input, val_target, val_recommend)


def main():
    logistic_regression(data_type='sample')


if __name__ == '__main__':
    main()
