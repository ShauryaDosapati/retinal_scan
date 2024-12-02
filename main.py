import os
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

IMAGE_SIZE = [224, 224]
train_path = '/Users/shauryad/Developer/python/Datasets/RetinalScan/train'
test_path = '/Users/shauryad/Developer/python/Datasets/RetinalScan/test'

def combined_block(input_tensor, filters, kernel_size=3, padding="same", dropout=0.5):

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, activation=None)(
        input_tensor
    )
    shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, padding=padding, activation=None)(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size, padding=padding, activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size, padding=padding, activation=None)(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout)(x)

    return x

def build_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, padding="same", activation="relu")(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = combined_block(x, filters=64)
    x = combined_block(x, filters=128)
    x = combined_block(x, filters=256)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.7)(x)
    output_tensor = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

input_shape = (224, 224, 3)
num_classes = 5
model = build_model(input_shape, num_classes)

optimizer = Adam(learning_rate=0.00001)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMAGE_SIZE, batch_size=64, class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=IMAGE_SIZE, batch_size=64, class_mode="categorical"
)

model.summary()
r = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=1,
    steps_per_epoch=len(train_generator),
    validation_steps=len(test_generator),
)
model.save('retinal.h5')
