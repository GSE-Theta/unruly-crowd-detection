import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold

f_abnormal = sorted(os.listdir('dataset/umn_dataset/abnormal'))
f_normal = sorted(os.listdir('dataset/umn_dataset/normal'))
n_abnormal = len(f_abnormal)
n_normal = len(f_normal)
y = np.concatenate([np.zeros(n_abnormal), np.ones(n_normal)])
f = f_abnormal + f_normal
n = len(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(n_abnormal, n_normal)

shutil.rmtree('dataset/k-fold-validation', ignore_errors=True)

for i, (train_index, test_index) in enumerate(skf.split(np.zeros(n), y)):
    # select rows
    train_y, test_y = y[train_index], y[test_index]
    # summarize train and test composition
    train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
    test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
    print('Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))

    os.makedirs('dataset/k-fold-validation', exist_ok=True)
    os.makedirs('dataset/k-fold-validation/fold-%d' % (i+1))
    os.makedirs('dataset/k-fold-validation/fold-%d/train' % (i+1))
    os.makedirs('dataset/k-fold-validation/fold-%d/valid' % (i+1))
    os.makedirs('dataset/k-fold-validation/fold-%d/train/abnormal' % (i+1))
    os.makedirs('dataset/k-fold-validation/fold-%d/train/normal' % (i+1))
    os.makedirs('dataset/k-fold-validation/fold-%d/valid/abnormal' % (i+1))
    os.makedirs('dataset/k-fold-validation/fold-%d/valid/normal' % (i+1))
    for j in train_index:
        c = 'abnormal' if y[j] == 0 else 'normal'
        # print('dataset/spatial/%s/%s' % (c, f[j]))
        # print('dataset/fold-%d/train/%s/%s' % (i+1, c, f[j]))
        shutil.copyfile('dataset/umn_dataset/%s/%s' % (c, f[j]), 'dataset/k-fold-validation/fold-%d/train/%s/%s' % (i+1, c, f[j]))
    for j in test_index:
        c = 'abnormal' if y[j] == 0 else 'normal'
        # print('dataset/spatial/%s/%s' % (c, f[j]))
        # print('dataset/fold-%d/test/%s/%s' % (i+1, c, f[j]))
        shutil.copyfile('dataset/umn_dataset/%s/%s' % (c, f[j]), 'dataset/k-fold-validation/fold-%d/valid/%s/%s' % (i+1, c, f[j]))
