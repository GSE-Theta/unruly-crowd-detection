import tensorflow as tf
from scipy.stats import ttest_1samp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from model import preprocess_input, get_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mean', type=float, required=True)
args = parser.parse_args()

samples = []
model = get_model()
model.summary()

for data in ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5']:
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset/%s/valid' % data,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=16,
        image_size=(214, 320)
    )
    valid_ds = valid_ds.map(preprocess_input).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    model.load_weights('model/%s_weights.h5' % data)

    y_true = tf.concat([y for x, y in valid_ds], axis=0)
    y_pred = model.predict(valid_ds)
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print('tn: %d, fp: %d, fn: %d, tp: %d' % (tn, fp, fn, tp))
    print('precision: %f, recall: %f, f1_score: %f' % (precision, recall, f1_score))
    samples.append(f1_score)
    del valid_ds, y_true, y_pred

t_test = ttest_1samp(samples, popmean=args.mean, alternative="less")
print('t-statistic: %f, p-value: %f' % (t_test.statistic, t_test.pvalue))
