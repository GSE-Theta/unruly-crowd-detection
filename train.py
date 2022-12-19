import tensorflow as tf
from model import compressed_xception
import argparse

def preprocess_input(x, y):
    return tf.keras.applications.xception.preprocess_input(x), y

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/%s/train' % args.data,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=16,
    image_size=(214, 320)
)

train_ds = train_ds.map(preprocess_input).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    compressed_xception,
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2)
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
)
model.fit(train_ds, epochs=10)

model.save('model/%s' % args.data)
