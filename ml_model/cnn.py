import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('pcos_ultrasound/training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255) 
test_set = test_datagen.flow_from_directory('pcos_ultrasound/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn.fit(x=training_set,validation_data=test_set,epochs=25)

test_image = image.load_img('pcos_ultrasound/single_prediction/img4.jpg', target_size = (64, 64))   #Infected
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0 
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] >0.5:
  prediction = 'not_infected'
else:
  prediction = 'infected'
print(prediction)

# Save the trained model
cnn.save('cnn_model.keras')
