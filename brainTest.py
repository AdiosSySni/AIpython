# %%
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# Параметры для обработки изображений
img_height = 224
img_width = 224
batch_size = 64

# %%
dataset_dir = './dataset'
classes = os.listdir(dataset_dir)

print(classes)

# %%
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
    './dataset',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    './validDataset/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)


print(train_generator.class_indices)

# %%
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(train_generator.num_classes, activation='relu')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision'])

# %%
if os.path.exists('newModel.keras'):
    model = tf.keras.models.load_model('newModel.keras')
else:
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=15,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

model.save('newModel.keras')

# %%
model_load = tf.keras.models.load_model('newModel.keras')

loss, acc = model.evaluate(validation_generator, verbose=2)
print("Validation accuracy:", acc)

# %%
def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_batch)

# Загружаем и подготавливаем изображение
new_image = load_and_preprocess_image('Salix_alba_Morton.jpg')

# %%
# Выполняем предсказание
preds = model.predict(new_image)
predicted_class_index = np.argmax(preds)

# Получаем имя класса на основе индекса
predicted_class_name = classes[predicted_class_index]

print("Predicted class:", predicted_class_name)

# %%
def find_similar_images(class_index, n=10):
    similar_images = []
    # Заменить class_index на class_name
    for root, dirs, files in os.walk(f'dataset/{class_index}'):
        for file in files[:n]:
            image_path = os.path.join(root, file)
            similar_images.append(image_path)
            print(image_path)

    print(similar_images)        
    return similar_images

# %%
similar_images = find_similar_images(predicted_class_name)

# %%



