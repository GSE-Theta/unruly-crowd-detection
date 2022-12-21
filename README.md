GSE-Theta - Unruly Behavior Detection
===================================================

How To
--------
First create `dataset` directory and put your dataset into it:
```
mkdir dataset
```

Create `model` directory:
```
mkdir model
```

Run `split.py`, this will split your dataset into 5-fold for cross-validation:
```
# Please make sure that the amount of data can divide by 5
python split.py
```

Run `train_all.py` for train 5 models consecutively, or `train.py` to spectific the fold of data to train:
```
python train_all.py
# or
python train.py --data fold-1
```

Run `t_test.py` for evaluating the trained model and calculating the p-value with null hypothesis (mu <= 0.95):
```
python t_test.py --mean 0.95
```

Also, you can run `eval.py` for evaluating specific model:
```
python eval.py --data fold-1
```

You can also put the desired video into `video` directory and run `video.py` (`out` is optional):
```
python video.py --model fold-1 --in Crowd-Activity-All.avi --out output.avi
```
