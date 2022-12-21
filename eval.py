from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.utils import image_dataset_from_directory
from tensorflow import data, concat, argmax
from sklearn.metrics import confusion_matrix, classification_report
from model import preprocess_input, get_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

valid_ds = image_dataset_from_directory(
    'dataset/k-fold-validation/%s/valid' % args.data,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=16,
    image_size=(214, 320)
)
valid_ds = valid_ds.map(preprocess_input).cache().prefetch(buffer_size=data.AUTOTUNE)

model = get_model()
model.load_weights('model/%s_weights.h5' % args.data)
model.compile(loss=CategoricalCrossentropy(from_logits=True), metrics=[CategoricalAccuracy(name='accuracy')])

y_true = concat([y for _, y in valid_ds], axis=0)
y_pred = model.predict(valid_ds)
y_true = argmax(y_true, axis=1)
y_pred = argmax(y_pred, axis=1)

tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
print('tp: %d, fn: %d, fp: %d, tn: %d' % (tp, fn, fp, tn))
print(classification_report(y_true, y_pred, target_names=['abnormal', 'normal'], digits=4))
