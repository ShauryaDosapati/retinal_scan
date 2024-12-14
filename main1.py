import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.src.applications import ResNet101, VGG19
from keras.src.layers import Flatten
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Add,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.regularizers import l2

print("TensorFlow Version:", tf.__version__)

IMAGE_SIZE = [224, 224]
train_path = '/Users/shauryad/Developer/python/sia-gen-ai-services/RetinalScan/train'
test_path = '/Users/shauryad/Developer/python/sia-gen-ai-services/RetinalScan/test'


vgg19 = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg19.layers:
    layer.trainable = False
x = vgg19.output
x = Flatten()(vgg19.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(5,activation='softmax')(x)
model = Model(inputs=vgg19.input, outputs=x)


optimizer = Adam(learning_rate=1e-4)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy",tf.keras.metrics.Precision(),tf.keras.metrics.Recall()],
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMAGE_SIZE, batch_size=32, class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=IMAGE_SIZE, batch_size=32, class_mode="categorical"
)
class_labels = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(class_labels), y=class_labels
)
class_weights = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weights)


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1
)

model.summary()
r = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    steps_per_epoch=len(train_generator),
    validation_steps=len(test_generator),
)
model.save('retinal.h5')
