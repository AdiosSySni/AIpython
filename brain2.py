import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import matplotlib.pyplot as plt

img_height = 224
img_width = 224
batch_size = 32

dataset_dir = './dataset'
classes = os.listdir(dataset_dir)

print(classes)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

validation_generator = test_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print(train_generator.class_indices)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if not os.path.exists('newModel.keras'):
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=5,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )
    model.save('newModel.keras')
else:
    model = tf.keras.models.load_model('newModel.keras')


model_load = tf.keras.models.load_model('newModel.keras')

loss, acc = model.evaluate(validation_generator, verbose=)
print("Validation accuracy:", acc)


feature_extractor = Model(inputs=model.inputs, outputs=x)

def extract_features(generator):
    features = []
    labels = []
    for inputs, targets in generator:
        batch_features = feature_extractor.predict(inputs)
        features.extend(batch_features)
        labels.extend(targets)
    return np.array(features), np.array(labels)

train_features, train_labels = extract_features(train_generator)
val_features, val_labels = extract_features(validation_generator)

np.savez('train_features.npz', features=train_features, labels=train_labels)
np.savez('val_features.npz', features=val_features, labels=val_labels)

def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_batch)

def find_similar_images(query_image, k=10):
    query_features = feature_extractor.predict(load_and_preprocess_image(query_image))[0]
    
    distances = np.linalg.norm(val_features - query_features, axis=1)
    indices = np.argsort(distances)[:k]
    
    similar_images = [(f"image_{i}.jpg", d) for i, d in zip(indices, distances[indices])]
    return similar_images

query_image_path = 'Salix_alba_Morton.jpg'
similar_images = find_similar_images(query_image_path)

for image, distance in similar_images:
    print(f"Image: {image}, Distance: {distance:.4f}")