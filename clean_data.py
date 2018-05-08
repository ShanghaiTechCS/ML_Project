# -*- coding: utf-8 -*-
# @Time    : 2018/5/7 下午8:15
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import csv
import numpy as np
import pickle

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


def format_attr(attr_array, name, fill_none='zero', normalization=True):
    """
    :param attr: attr_array [N, 1]
    :param name: attr name
    :param fill_none: ['average', 'sample', 'zero']
    :return formatted_attr_array [N, C] C is new attr number
    """

    def fill_none_attr(attr_array, filled_attr):
        na_list = ['NA', 'unknown']
        for na_attr in na_list:
            attr_array[attr_array == na_attr] = filled_attr

    attr_array = attr_array.copy()
    attr_type = attr_type_dict[name]
    filled_attr = None
    standard_weight_list = []

    # string
    if attr_type == 'enum':
        if fill_none == 'zero':
            filled_attr = ''
        fill_none_attr(attr_array, filled_attr)

        unique_attr_array = np.unique(attr_array)
        formatted_attr_array = []

        for unique_attr in unique_attr_array:
            formatted_attr_array.append(attr_array == unique_attr)
            standard_weight_list.append((None, None))
        formatted_attr_array = np.concatenate(formatted_attr_array, axis=1)


    # int or float
    else:
        if fill_none == 'zero':
            filled_attr = 0

        fill_none_attr(attr_array, filled_attr)
        bool_name_dict = {'yes': 1, 'no': 0}
        for bool_name, bool_value in bool_name_dict.items():
            attr_array[attr_array == bool_name] = bool_value

        formatted_attr_array = attr_array.astype(np.float32)
        if normalization:
            mean, std = np.mean(formatted_attr_array), np.std(formatted_attr_array)
            formatted_attr_array = (formatted_attr_array - mean) / std
            standard_weight_list.append((mean, std))

    if normalization:
        return formatted_attr_array, standard_weight_list
    else:
        return formatted_attr_array


def clean_train_customer(src_data_path, dest_data_path):
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
    feature_array = np.delete(feature_array, -2, axis=1) #
    target_array = data_array[:, -3:-1]  # str [N, 2]

    print('feature_array shape:', feature_array.shape)
    print('target_array shape:', target_array.shape)
    print('')
    sample_num, feature_num = feature_array.shape

    # formatted feature array
    formatted_feature_array = []
    feature_standard_weight_list = []
    for i in range(feature_num):
        formatted_attr_array, standard_weight_list = format_attr(feature_array[:, i:i + 1], feature_attr_list[i],
                                                                 fill_none='zero', normalization=True)  # (N, C)
        formatted_feature_array.append(formatted_attr_array)
        feature_standard_weight_list += standard_weight_list
    formatted_feature_array = np.concatenate(formatted_feature_array, axis=1)  # (N, FF)

    # formatted target array (just change str to float)
    responded_target = format_attr(target_array[:, 0:1], target_attr_list[0], fill_none='zero', normalization=False)
    profit_target = format_attr(target_array[:, 1:2], target_attr_list[1], fill_none='zero', normalization=False)

    responded_idx = np.argwhere(responded_target)[:, 0]
    only_profit_target = profit_target[responded_idx]
    wished_target = ((responded_target == 1) * (profit_target > 30)).astype(np.float32)

    data_package = {'feature_standard_weight_list': feature_standard_weight_list,
                    'responded_input': formatted_feature_array,
                    'responded_target': responded_target,
                    'wished_target': wished_target,
                    'profit_input': formatted_feature_array[responded_idx],
                    'profit_target': only_profit_target}

    for k, v in data_package.items():
        if isinstance(v, list):
            print('%s: %s' % (k, len(v)))
        else:
            print('%s: %s' % (k, v.shape))

    with open(dest_data_path, 'wb') as f:
        pickle.dump(data_package, f)


def main():
    clean_train_customer(src_data_path='data/DataTraining.csv', dest_data_path='data/zero/train.data')


if __name__ == '__main__':
    main()
