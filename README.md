# ML_Project

**feature_attr_list**: ['custAge', 'profession', 'marital', 'schooling', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'pastEmail']

**target_attr_list**: ['responded', 'profit']

```
feature_array shape: (8137, 20)
target_array shape: (8137, 2)
```

data_package:
```
feature_standard_weight_list: 55
train_input: (7323, 55)
train_target: (7323, 2)
val_input: (814, 55)
val_target: (814, 2)
```

## baseline1

experiments | mode  | setup | acc_most| profit_most| figure_name
------------| ----  | ----- | --------| -----------| -----------
baseline1   | sample | 原始的cls labels, svm | 0.917(c=0.01), 4353 | 0.82, 7731(0.343), RR=0.684| acc_bs1_svm1.png, profit_bs1_svm1.png
baseline1   | sample | synthesis cls labels, svm | 0.925(c=0.01), 4353 | 0.845, 8103(0.343), RR=0.689| acc_bs1_svm2.png, profit_bs1_svm2.png
baseline1   | average | 原始的cls labels, svm | 0.917(c=0.01), 4353 | 0.82, 7109(0.364)| acc_average_bs1_svm1.png, profit_average_bs1_svm1.png
baseline1   | average | synthesis cls labels, svm | 0.925(c=0.01), 4353 | 0.845, 8110(0.333)| acc_average_bs1_svm2.png, profit_average_bs1_svm2.png


## Tree Method
TP: total_precision   
RR: recommend_recall

| Baseline | data mode |  setup  |         train_result         |         val_result          |
| :------: | :-------: | :-----: | :--------------------------: | :-------------------------: |
|    1     |   zero    | base DT | TP:0.95 RR:0.57 Profit:62244 | TP:0.89 RR:0.36 Profit:4473 |
|    1     |  average  | base DT | TP:0.95 RR:0.59 Profit:64868 | TP:0.90 RR:0.33 Profit:4408 |
|    1     |  sample   | base DT | TP:0.94 RR:0.45 Profit:49059 | TP:0.90 RR:0.34 Profit:4839 |

## LR and MLR Result
| Baseline | data mode |            setup            |         train_result         |         val_result          |
| :------: | :-------: | :-------------------------: | :--------------------------: | :-------------------------: |
|    1     |  sample   |  LR, L2 norm, balance loss  | TP:0.80 RR:0.74 Profit:47226 | TP:0.78 RR:0.71 Profit:5826 |
|    1     |  sample   |  LR, L2 norm, balance data  | TP:0.81 RR:0.72 Profit:47936 | TP:0.79 RR:0.68 Profit:5945 |
|    1     |  sample   | MLR2, L2 norm, balance loss | TP:0.81 RR:0.72 Profit:48880 | TP:0.79 RR:0.69 Profit:5946 |
|    1     |  sample   | MLR4, L2 norm, balance loss | TP:0.92 RR:0.99 Profit:96072 | TP:0.84 RR:0.47 Profit:4542 |