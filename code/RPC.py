import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_dir = r'D:\PycharmProjects\deeplearning_developer\datasets\rps'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

sum_train = 0
for i in os.listdir(os.path.join(base_dir, 'train')):
    sum_train += len(os.listdir(os.path.join(base_dir, 'train',i)))
print('sum_train: ' + str(sum_train))

sum_validation = 0
for i in os.listdir(os.path.join(base_dir, 'validation')):
    sum_validation += len(os.listdir(os.path.join(base_dir, 'validation',i)))
print('sum_validation: ' + str(sum_validation))

train_datagen = ImageDataGenerator(rescale= 1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest'
                                   )
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size= (150,150),
                                                   batch_size= 16,
                                                   class_mode= 'categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                          target_size= (150,150),
                                                              batch_size=8,
                                                          class_mode= 'categorical')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss= 'categorical_crossentropy',
              optimizer= RMSprop(lr=0.001),
              metrics=['accuracy']
              )

history = model.fit(train_generator,
                    epochs=10, #保证batch_size（图像增强中）*steps_per_epoch（fit中）小于等于训练样本数
                    steps_per_epoch=140, #2520 images = batch_size * steps 保证batch_size（图像增强中）*steps_per_epoch（fit中）小于等于训练样本数
                    validation_data=validation_generator,
                    validation_steps=15,  #1000 images = batch_size * steps
                    verbose=2)

model.save(os.path.join(base_dir,'rps.h5'))

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()

restor_model = tf.keras.models.load_model(os.path.join(base_dir,'rps.h5'))

restor_model.summary()

import numpy as np
from keras.preprocessing import image

for i in os.listdir(os.path.join(base_dir,'test')):
    img = image.load_img(os.path.join(base_dir,'test',i), target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=16)
    print(i,classes)