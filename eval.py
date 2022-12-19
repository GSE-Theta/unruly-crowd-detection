import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import argparse

def preprocess_input(x, y):
    return tf.keras.applications.xception.preprocess_input(x), y

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/%s/valid' % args.data,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=16,
    image_size=(214, 320)
)
valid_ds = valid_ds.map(preprocess_input).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = tf.keras.models.load_model('model/%s' % args.data)
model.evaluate(valid_ds)

true_labels = tf.concat([y for x, y in valid_ds], axis=0)
predicted_labels = model.predict(valid_ds)
true_labels = tf.argmax(true_labels, axis=1)
predicted_labels = tf.argmax(predicted_labels, axis=1)

print(confusion_matrix(true_labels, predicted_labels))
print(classification_report(true_labels, predicted_labels, target_names=['abnormal', 'normal'], digits=4))