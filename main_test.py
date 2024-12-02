from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# Set image size and paths
IMAGE_SIZE = (224, 224)
valid_path = '/Users/shauryad/Developer/python/Datasets/RetinalScan/val'
img_path = '/Users/shauryad/Developer/python/Datasets/RetinalScan/test/Stage_2/e65a2ff90494.png'

# Load the trained model
model = load_model('retinal_scan.h5')

# Create a data generator for validation data
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
training_set = train_datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    class_mode='categorical',
)

# Map class indices to their corresponding labels
class_labels = training_set.class_indices
labels = {v: k for k, v in class_labels.items()}
print("Class Labels:", labels)

# Load and preprocess the image for prediction
img = image.load_img(img_path, target_size=IMAGE_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# Predict the class of the input image
classes = model.predict(x)
predicted_classes = np.argmax(classes, axis=1)[0]  # Extract the scalar value

# Map the predicted class to its corresponding label
result_label = labels[predicted_classes]

# Print the diagnosis based on the result label
if result_label == 'Stage_0':
    print("Diagnosis: No Diabetic Retinopathy (Stage 0)")
elif result_label == 'Stage_1':
    print("Diagnosis: Mild Diabetic Retinopathy (Stage 1)")
elif result_label == 'Stage_2':
    print("Diagnosis: Moderate Diabetic Retinopathy (Stage 2)")
elif result_label == 'Stage_3':
    print("Diagnosis: Severe Diabetic Retinopathy (Stage 3)")
elif result_label == 'Stage_4':
    print("Diagnosis: Proliferative Diabetic Retinopathy (Stage 4)")
else:
    print("Error: Unknown Stage")
