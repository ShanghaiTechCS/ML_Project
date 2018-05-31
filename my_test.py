# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午6:29
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import numpy as np
import pickle
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
import seaborn as sns
import csv


def main():
    with open('new_data/train.data', 'rb') as f:
        data_package = pickle.load(f)

    train_input = data_package['train_input'][:, ]
    train_target = data_package['train_target'][:, 0]

    train_num = train_input.shape[0]
    random_idx = random.sample(range(train_num), 1000)

    train_input = train_input[random_idx]
    train_target = train_target[random_idx]

    print(train_input.shape)
    print(train_target.shape)
    #
    # Cor = np.abs(np.corrcoef(train_input.T))
    #
    # print(np.sum(Cor > 0.9))
    # print(np.sum(Cor > 0.8))
    #
    # sns.set()
    # sns.heatmap(Cor, cmap="YlGnBu")
    # plt.show()

    X_embedded = TSNE(n_components=2).fit_transform(train_input)

    plt.scatter(X_embedded[train_target == 0][:, 0], X_embedded[train_target == 0][:, 1], marker='o')
    plt.scatter(X_embedded[train_target == 1][:, 0], X_embedded[train_target == 1][:, 1], marker='o')
    plt.show()

    train_input, train_target = SMOTE().fit_sample(train_input, train_target)

    X_embedded = TSNE(n_components=2).fit_transform(train_input)

    plt.scatter(X_embedded[train_target == 0][:, 0], X_embedded[train_target == 0][:, 1], marker='o')
    plt.scatter(X_embedded[train_target == 1][:, 0], X_embedded[train_target == 1][:, 1], marker='o')

    plt.show()


if __name__ == '__main__':
    main()


