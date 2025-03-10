import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('pcos_ultrasound/training',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('pcos_ultrasound/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

dataset = pd.read_csv('PCOS.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=25)

test_image = image.load_img('pcos_ultrasound/single_prediction/img6.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0 
test_image = np.expand_dims(test_image, axis = 0)
cnn_prediction = cnn.predict(test_image)
ann_prediction=ann.predict(sc.transform([[34, 47.9,1, 1,0, 1,1, 4.0, 15.0,3.0,20]]))
final_score = 0.6 * cnn_prediction + 0.4 * ann_prediction
if final_score >= 0.5:
    print("High chances of pcos")
else:
    print("PCOS not detected")