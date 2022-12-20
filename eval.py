from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.utils import image_dataset_from_directory
from tensorflow import data, concat, argmax
from sklearn.metrics import classification_report
from model import preprocess_input, get_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

valid_ds = image_dataset_from_directory(
    'dataset/%s/valid' % args.data,
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
model.evaluate(valid_ds)

true_labels = concat([y for _, y in valid_ds], axis=0)
predicted_labels = model.predict(valid_ds)
true_labels = argmax(true_labels, axis=1)
predicted_labels = argmax(predicted_labels, axis=1)

print(classification_report(true_labels, predicted_labels, target_names=['abnormal', 'normal'], digits=4))
