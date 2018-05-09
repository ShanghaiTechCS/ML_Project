import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import pdb
import cv2
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

    responded_synthesis = responded_target[7310:].squeeze().astype(np.int8) & (profit_target > 30).squeeze().astype(np.int8)
    responded_synthesis_target = np.concatenate((responded_target[:7310].squeeze(), responded_synthesis))
    training_data = []
    training_gt = []
    val_data = []
    val_gt = []

    data_dict = {}
    for i in range(length):

        seed = np.random.random()
        if seed < 0.8:
            training_data.append(responded_input[i])
            training_gt.append(responded_synthesis_target[i])
        else:
            val_data.append(responded_input[i])
            val_gt.append(responded_synthesis_target[i])

    import pdb
    pdb.set_trace()
    data_dict['training_data'] = np.array(training_data, dtype=np.float64)
    data_dict['training_gt'] = np.array(training_gt, dtype=np.float64)
    data_dict['val_data'] = np.array(val_data, dtype=np.float64)
    data_dict['val_gt'] = np.array(val_gt, dtype=np.float64)

    return data_dict

def baseline1(data_dict=None):
    """
    This baseline can estimate the customer whether responded.
    And the groundtruth is the (responded_target \cap (profit_target>30))

    :param data_dict:
    :return:
    """


def main():
    print(123)


if __name__ == '__main__':
    # main()

    dataloder(path='./data/zero')
