from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.utils import image_dataset_from_directory
from tensorflow import data
from model import preprocess_input, get_model
import os

for fold in sorted(os.listdir('dataset/k-fold-validation')):
    train_ds = image_dataset_from_directory(
        'dataset/k-fold-validation/%s/train' % fold,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=16,
        image_size=(214, 320)
    )
    train_ds = train_ds.map(preprocess_input).cache().prefetch(buffer_size=data.AUTOTUNE)

    model = get_model()
    model.summary()

    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(from_logits=True), metrics=[CategoricalAccuracy(name='accuracy')])
    model.fit(train_ds, epochs=20)

    model.save_weights('model/%s_weights.h5' % fold)
    del train_ds
