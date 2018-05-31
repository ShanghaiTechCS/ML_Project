# -*- coding: utf-8 -*-
# @Time    : 2018/5/31 下午2:12
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import csv
import numpy as np
import json
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

attr_type_dict = {
    'custAge': 'int',
    'profession': 'enum',
    'marital': 'enum',
    'schooling': 'enum',
    'default': 'bool',
    'housing': 'bool',
    'loan': 'bool',
    'contact': 'enum',
    'month': 'enum',
    'day_of_week': 'enum',
    'campaign': 'int',
    'pdays': 'int',
    'previous': 'int',
    'poutcome': 'enum',
    'emp.var.rate': 'float',
    'cons.price.idx': 'float',
    'cons.conf.idx': 'float',
    'euribor3m': 'float',
    'nr.employed': 'float',
    'pmonths': 'int',
    'pastEmail': 'int',
    'responded': 'bool',
    'profit': 'int'
}

customer_attr_name_list = ['custAge', 'profession', 'marital', 'schooling', 'housing', 'loan', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
                           'nr.employed']


def format_attr(attr_array, name, normalization=True):
    """
    :param attr: attr_array [N, 1]
    :param name: attr name
    :param fill_none: 'average', 'sample', 'zero'
    :return formatted_attr_array [N, C] C is new attr number
    """

    def deal_num():
        bool_name_dict = {'yes': 1, 'no': 0}
        for bool_name, bool_value in bool_name_dict.items():
            attr_array[attr_array == bool_name] = bool_value

        formatted_attr_array = attr_array.astype(np.float32)
        if normalization:
            mean, std = np.mean(formatted_attr_array), np.std(formatted_attr_array)
            formatted_attr_array = (formatted_attr_array - mean) / std
            standard_weight_list.append((mean, std))

        return formatted_attr_array

    def deal_str():
        unique_attr_array = np.unique(attr_array)
        formatted_attr_array = []

        for unique_attr in unique_attr_array:
            formatted_attr_array.append(attr_array == unique_attr)
            standard_weight_list.append((None, None))
        formatted_attr_array = np.concatenate(formatted_attr_array, axis=1)

        return formatted_attr_array

    attr_array = attr_array.copy()
    attr_type = attr_type_dict[name]
    standard_weight_list = []

    # string
    if attr_type == 'enum':
        formatted_attr_array = deal_str()
    # int or float or bool
    else:
        formatted_attr_array = deal_num()

    if normalization:
        return formatted_attr_array, standard_weight_list
    else:
        return formatted_attr_array


def count_NA_data(src_data_path):
    na_list = ['NA', 'unknown']

    with open(src_data_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    title_list = data_list[0]
    target_attr_list = title_list[-2:]
    feature_attr_list = title_list[:-2]

    print('feature_attr_list:', feature_attr_list)
    print('target_attr_list:', target_attr_list)

    data_array = np.array(data_list[1:])  # (N, F+2)
    feature_array = data_array[:, :-2]  # str [N, F]
    target_array = data_array[:, -2:]  # str [N, 2]

    print('feature_array shape:', feature_array.shape)
    print('target_array shape:', target_array.shape)
    print()

    sample_num, feature_num = feature_array.shape
    for i in range(feature_num):
        print(feature_attr_list[i], np.sum(feature_array[:, i] == 'NA'), np.sum(feature_array[:, i] == 'unknown'))

    filled_feat_array = fill_data_by_nearest_neighbor(feature_array, feature_attr_list)

    # save
    with open('new2_data/filled_new2_DataTraining.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(feature_attr_list)
        writer.writerows(filled_feat_array)


def fill_data_by_nearest_neighbor(feature_array, feat_name_list):
    def get_dist(feat1, feat2, feat_name_list, feat_max_min_list):
        """
        :param feat1: (feat_dim, )
        :param feat2: (feat_dim, )
        :param feat_name_list: list (feat_dim, )
        :param feat_max_min_list: list (feat_dim, 2)
        :return:
        """
        feat_dim = feat1.shape[0]

        dist = 0
        for i in range(feat_dim):
            if feat1[i] == 'NA' or feat2[i] == 'NA':
                dist += 1
            elif attr_type_dict[feat_name_list[i]] in ['int', 'float']:
                max_v, min_v = feat_max_min_list[i]
                norm_feat1_i = (float(feat1[i]) - min_v) / (max_v - min_v)
                norm_feat2_i = (float(feat2[i]) - min_v) / (max_v - min_v)

                dist += (norm_feat1_i - norm_feat2_i) ** 2
            elif feat1[i] != feat2[i]:
                dist += 1

        dist = np.sqrt(dist)
        return dist

    na_list = ['NA', 'unknown']
    feature_array = feature_array.copy()

    # fill all in 'NA'
    for na in na_list:
        feature_array[feature_array == na] = 'NA'

    sample_num, feat_dim = feature_array.shape
    feat_max_min_list = []
    for feat_idx in range(feat_dim):
        if attr_type_dict[feat_name_list[feat_idx]] in ['int', 'float']:
            cur_feat_array = feature_array[:, feat_idx]
            cur_feat_array = cur_feat_array[cur_feat_array != 'NA'].astype(np.float)

            feat_max_min_list.append([cur_feat_array.max(), cur_feat_array.min()])
        else:
            feat_max_min_list.append([0, 0])

    for feat1_idx in range(sample_num):
        feat1 = feature_array[feat1_idx]
        if (feat1 == 'NA').sum() != 0:

            feat_dist_list = []
            for feat2_idx in range(sample_num):
                if feat1_idx != feat2_idx:
                    feat2 = feature_array[feat2_idx]
                    dist = get_dist(feat1, feat2, feat_name_list, feat_max_min_list)
                    feat_dist_list.append((feat2_idx, dist))

            feat_dist_list = sorted(feat_dist_list, key=lambda x: x[1])

            na_idx_list = np.argwhere(feat1 == 'NA').reshape(-1).tolist()
            na_idx_set = set(na_idx_list)

            for feat2_idx, dist in feat_dist_list:
                if len(na_idx_set) == 0:
                    break

                feat2 = feature_array[feat2_idx]
                removed_na_idx_set = set()
                for na_idx in na_idx_set:
                    if feat2[na_idx] != 'NA':
                        feature_array[feat1_idx, na_idx] = feat2[na_idx]
                        removed_na_idx_set.add(na_idx)
                na_idx_set -= removed_na_idx_set
        print('%d ok!' % feat1_idx)

    return feature_array


def normalization(src_data_path):
    with open(src_data_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    title_list = data_list[0]
    target_attr_list = title_list[-2:]
    feature_attr_list = title_list[:-2]

    print('feature_attr_list:', feature_attr_list)
    print('target_attr_list:', target_attr_list)

    data_array = np.array(data_list[1:])  # (N, F+2)
    feature_array = data_array[:, :-2]  # str [N, F]
    target_array = data_array[:, -2:]  # str [N, 2]

    print('feature_array shape:', feature_array.shape)
    print('target_array shape:', target_array.shape)
    print()

    # formatted feature array
    formatted_feature_array = []
    formatted_feature_name_list = []
    feature_standard_weight_list = []
    sample_num, feature_num = feature_array.shape
    for i in range(feature_num):
        formatted_attr_array, standard_weight_list = format_attr(feature_array[:, i:i + 1], feature_attr_list[i], normalization=True)  # (N, C)
        new_feat_dim = formatted_attr_array.shape[1]

        if new_feat_dim == 1:
            formatted_feature_name_list.append(feature_attr_list[i])
        else:
            formatted_feature_name_list += ['%s#%d' % (feature_attr_list[i], k) for k in range(new_feat_dim)]

        formatted_feature_array.append(formatted_attr_array)
        feature_standard_weight_list += standard_weight_list
    formatted_feature_array = np.concatenate(formatted_feature_array, axis=1)  # (N, FF)

    # formatted target array (just change str to float)
    responded_target = format_attr(target_array[:, 0:1], target_attr_list[0], normalization=False)
    profit_target = format_attr(target_array[:, 1:2], target_attr_list[1], normalization=False)

    # denote
    input_data = formatted_feature_array  # (N, C)
    target_data = np.concatenate((responded_target, profit_target), axis=1)  # (N, 2)

    print('input_data shape:', input_data.shape)
    print('target_data shape:', target_data.shape)

    # remove_related_feature(input_data, formatted_feature_name_list)

    total_data = np.concatenate((input_data, target_data), axis=1)

    # save
    with open('new2_data/normalized_filled_new2_DataTraining.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(formatted_feature_name_list + target_attr_list)
        writer.writerows(total_data)

    with open('new2_data/feature_standard_weight_list.pkl', 'wb') as f:
        pickle.dump(feature_standard_weight_list, f)


def remove_related_feature(input_data, feature_name_list):
    Cor = np.abs(np.corrcoef(input_data.T))
    sns.set()
    sns.heatmap(Cor, cmap="YlGnBu")
    plt.show()

    np.fill_diagonal(Cor, 0)
    print(np.sum(Cor > 0.9))

    feat_idx = np.argwhere(Cor > 0.9)
    for idx1, idx2 in feat_idx:
        print(idx1, idx2, Cor[idx1, idx2])

    unique_feat_idx = np.unique(feat_idx.reshape(-1))
    for fid in unique_feat_idx:
        print(fid, feature_name_list[fid])

    exit()


def final_new_data(src_data_path, feature_standard_weight_list_path):
    with open(src_data_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    title_list = data_list[0]
    target_name_list = title_list[-2:]
    feat_name_list = title_list[:-2]

    data_array = np.array(data_list[1:]).astype(np.float)  # (N, F+2)
    input_data = data_array[:, :-2]  # str [N, F]
    target_data = data_array[:, -2:]  # str [N, 2]

    print('input_data shape:', input_data.shape)
    print('target_data shape:', target_data.shape)
    print()

    with open(feature_standard_weight_list_path, 'rb') as f:
        feature_standard_weight_list = pickle.load(f)

    # divide in train set and val set
    with open('data/train_val_list.json', 'r') as f:
        train_val_idx = json.load(f)
        train_idx_list, val_idx_list = train_val_idx['train_idx_list'], train_val_idx['val_idx_list']

    train_input = input_data[train_idx_list, :]
    train_target = target_data[train_idx_list, :]

    val_input = input_data[val_idx_list, :]
    val_target = target_data[val_idx_list, :]

    data_package = {'feature_standard_weight_list': feature_standard_weight_list,
                    'feat_name_list': feat_name_list,
                    'target_name_list': target_name_list,

                    'train_input': train_input,
                    'train_target': train_target,

                    'val_input': val_input,
                    'val_target': val_target}

    for k, v in data_package.items():
        if isinstance(v, list):
            print('%s: %s' % (k, len(v)))
        else:
            print('%s: %s' % (k, v.shape))

    # save in pkl
    with open('new2_data/train.data', 'wb') as f:
        pickle.dump(data_package, f)


def update_feature_standard_weight_list():
    delete_idx = [25, 43, 47]
    with open('new_data/feature_standard_weight_list.pkl', 'rb') as f:
        feature_standard_weight_list = pickle.load(f)
        new_idx_list = list(set(range(len(feature_standard_weight_list))) - set(delete_idx))

        feature_standard_weight_list = [feature_standard_weight_list[i] for i in new_idx_list]

    with open('new_data/feature_standard_weight_list.pkl', 'wb') as f:
        pickle.dump(feature_standard_weight_list, f)


def main():
    # fill_none = 'sample'
    # clean_train_customer(src_data_path='data/DataTraining.csv', dest_data_path='data/%s/train.data' % fill_none, fill_none=fill_none)

    # create_train_val_list('data/train_val_list.json')

    # count_NA_data(src_data_path='new2_data/new2_DataTraining.csv')
    # normalization(src_data_path='new2_data/filled_new2_DataTraining.csv')

    # update_feature_standard_weight_list()
    final_new_data('new2_data/final_new2_DataTraining.csv', 'new2_data/feature_standard_weight_list.pkl')
    pass


if __name__ == '__main__':
    main()
