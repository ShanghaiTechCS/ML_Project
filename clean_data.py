# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:15
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import csv
import numpy as np
import json
import random
import pickle
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


def format_attr(attr_array, name, fill_none, normalization=True):
    """
    :param attr: attr_array [N, 1]
    :param name: attr name
    :param fill_none: 'average', 'sample', 'zero'
    :return formatted_attr_array [N, C] C is new attr number
    """

    def fill_attr_array(attr_array, filled_attr):
        if filled_attr is None:
            exist_attr_array = attr_array[attr_array != 'NA']
            NA_num = (attr_array == 'NA').sum()
            filled_attr_array = np.random.choice(exist_attr_array, NA_num)
            attr_array[attr_array == 'NA'] = filled_attr_array
        else:
            attr_array[attr_array == 'NA'] = filled_attr

    def deal_num():
        if fill_none == 'zero':
            filled_attr = 0
        elif fill_none == 'average' and attr_type != 'bool':
            filled_attr = attr_array[attr_array != 'NA'].astype(np.float32).mean()
        else:
            filled_attr = None

        bool_name_dict = {'yes': 1, 'no': 0}
        for bool_name, bool_value in bool_name_dict.items():
            attr_array[attr_array == bool_name] = bool_value

        fill_attr_array(attr_array, filled_attr)
        formatted_attr_array = attr_array.astype(np.float32)
        if normalization:
            mean, std = np.mean(formatted_attr_array), np.std(formatted_attr_array)
            formatted_attr_array = (formatted_attr_array - mean) / std
            standard_weight_list.append((mean, std))

        return formatted_attr_array

    def deal_str():
        if fill_none == 'zero':
            filled_attr = ''
        elif fill_none == 'average':
            attr_array_without_NA = np.delete(attr_array, np.argwhere(attr_array == 'NA'))
            unique_value, unique_count = np.unique(attr_array_without_NA, return_counts=True)
            max_idx = unique_count.argmax()
            filled_attr = unique_value[max_idx]
        else:
            filled_attr = None

        fill_attr_array(attr_array, filled_attr)
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
    na_list = ['NA', 'unknown']

    # format NA
    for na_attr in na_list:
        attr_array[attr_array == na_attr] = 'NA'

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


def create_train_val_list(dest_data_path):
    total_num = 8137
    train_rate = 0.9
    train_num = int(train_rate * total_num)

    random_idx_list = random.sample(list(range(total_num)), total_num)
    train_idx_list = random_idx_list[:train_num]
    val_idx_list = random_idx_list[train_num:]

    with open(dest_data_path, 'w') as f:
        json.dump({'train_idx_list': train_idx_list, 'val_idx_list': val_idx_list}, f)


def load_and_clean_input_target_data(src_data_path, fill_none):
    """
    :param: src_data_path
    :param: fill_none: 'zero', 'average', 'sample'
    :return: feature_standard_weight_list
    :return: input_data (N, C)
    :return: target_data (N, 2)
    """
    with open(src_data_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    title_list = data_list[0]
    target_attr_list = title_list[-3:-1]
    feature_attr_list = title_list[:-3]
    feature_attr_list.remove(feature_attr_list[-2])

    print('feature_attr_list:', feature_attr_list)
    print('target_attr_list:', target_attr_list)

    data_array = np.array(data_list[1:])  # (N, F+2)
    feature_array = data_array[:, :-3]  # str [N, F]
    feature_array = np.delete(feature_array, -2, axis=1)  #
    target_array = data_array[:, -3:-1]  # str [N, 2]

    print('feature_array shape:', feature_array.shape)
    print('target_array shape:', target_array.shape)
    print('')
    sample_num, feature_num = feature_array.shape

    # formatted feature array
    formatted_feature_array = []
    customer_feature_array = []
    customer_feature_standard_weight_list = []
    feature_standard_weight_list = []
    for i in range(feature_num):
        formatted_attr_array, standard_weight_list = format_attr(feature_array[:, i:i + 1], feature_attr_list[i],
                                                                 fill_none=fill_none, normalization=True)  # (N, C)

        print(formatted_attr_array.shape)
        if feature_attr_list[i] in customer_attr_name_list:
            customer_feature_array.append(formatted_attr_array)
            customer_feature_standard_weight_list += standard_weight_list

        formatted_feature_array.append(formatted_attr_array)
        feature_standard_weight_list += standard_weight_list
    formatted_feature_array = np.concatenate(formatted_feature_array, axis=1)  # (N, FF)
    customer_feature_array = np.concatenate(customer_feature_array, axis=1)  # (N, CustF)

    # formatted target array (just change str to float)
    responded_target = format_attr(target_array[:, 0:1], target_attr_list[0], fill_none='zero', normalization=False)
    profit_target = format_attr(target_array[:, 1:2], target_attr_list[1], fill_none='zero', normalization=False)

    # denote
    input_data = formatted_feature_array  # (N, C)
    cust_data = customer_feature_array  # (N, Cust)
    target_data = np.concatenate((responded_target, profit_target), axis=1)  # (N, 2)

    return feature_standard_weight_list, input_data, target_data, cust_data, customer_feature_standard_weight_list


def clean_train_customer(src_data_path, dest_data_path, fill_none='zero'):
    # load and clean data
    feature_standard_weight_list, input_data, target_data, cust_data, customer_feature_standard_weight_list = load_and_clean_input_target_data(
        src_data_path, fill_none)

    # divide in train set and val set
    with open('data/train_val_list.json', 'r') as f:
        train_val_idx = json.load(f)
        train_idx_list, val_idx_list = train_val_idx['train_idx_list'], train_val_idx['val_idx_list']

    train_input = input_data[train_idx_list, :]
    train_cust = cust_data[train_idx_list, :]
    train_target = target_data[train_idx_list, :]

    val_input = input_data[val_idx_list, :]
    val_cust = cust_data[val_idx_list, :]
    val_target = target_data[val_idx_list, :]

    data_package = {'feature_standard_weight_list': feature_standard_weight_list,
                    'customer_feature_standard_weight_list': customer_feature_standard_weight_list,

                    'train_input': train_input,
                    'train_cust': train_cust,
                    'train_target': train_target,

                    'val_input': val_input,
                    'val_cust': val_cust,
                    'val_target': val_target}

    for k, v in data_package.items():
        if isinstance(v, list):
            print('%s: %s' % (k, len(v)))
        else:
            print('%s: %s' % (k, v.shape))

    with open(dest_data_path, 'wb') as f:
        pickle.dump(data_package, f)


def count_NA_data(src_data_path):
    na_list = ['NA', 'unknown']

    with open(src_data_path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    title_list = data_list[0]
    target_attr_list = title_list[-3:-1]
    feature_attr_list = title_list[:-3]
    feature_attr_list.remove(feature_attr_list[-2])

    print('feature_attr_list:', feature_attr_list)
    print('target_attr_list:', target_attr_list)

    data_array = np.array(data_list[1:])  # (N, F+2)
    feature_array = data_array[:, :-3]  # str [N, F]
    feature_array = np.delete(feature_array, -2, axis=1)  #
    target_array = data_array[:, -3:-1]  # str [N, 2]

    print('feature_array shape:', feature_array.shape)
    print('target_array shape:', target_array.shape)
    print()

    sample_num, feature_num = feature_array.shape

    for i in range(feature_num):
        print(feature_attr_list[i], 'NA:', np.sum(feature_array[:, i] == 'NA'), 'unknown:', np.sum(feature_array[:, i] == 'unknown'))


def main():
    # fill_none = 'sample'
    # clean_train_customer(src_data_path='data/DataTraining.csv', dest_data_path='data/%s/train.data' % fill_none, fill_none=fill_none)

    # create_train_val_list('data/train_val_list.json')

    count_NA_data(src_data_path='data/DataTraining.csv')

    pass


if __name__ == '__main__':
    main()
