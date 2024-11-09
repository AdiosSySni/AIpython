import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# Параметры для обработки изображений
img_height = 224
img_width = 224
batch_size = 32

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
    './dataset/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Создаем модель на базе ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if os.path.exists('newModel.keras'):
    model = tf.keras.models.load_model('newModel.keras')
else:
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=1,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )

model.save('newModel.keras')

# Оценка модели
loss, acc = model.evaluate(validation_generator, verbose=2)
print("Validation accuracy:", acc)

# Функция для поиска похожих изображений
def find_similar_images(class_index, n=5):
    similar_images = []
    
    for root, dirs, files in os.walk(f'dataset/{class_index}'):
        for file in files[:n]:
            image_path = os.path.join(root, file)
            similar_images.append(image_path)
            
    return similar_images

# Загружаем и обрабатываем новое изображение
def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_batch)

# Получаем предсказания для нового изображения
new_image = load_and_preprocess_image('./ship.jpg')
preds = model.predict(new_image)
predicted_class = np.argmax(preds)

# Находим самые похожие изображения
similar_images = find_similar_images(predicted_class)

# Визуализируем результаты
# plt.figure(figsize=(15, 7))

print(similar_images)

# for i, path in enumerate(similar_images):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(plt.imread(path))
#     plt.title(os.path.basename(path))
#     plt.axis('off')

# plt.show()