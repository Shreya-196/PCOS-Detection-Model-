import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

dataset=pd.read_csv('PCOS.csv')
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=4,activation='relu'))
ann.add(tf.keras.layers.Dense(units=4,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(X_train,y_train,batch_size=32,epochs=25)

print(ann.predict(sc.transform([[34, 47.9,1, 1,0, 1,1, 4.0, 15.0,3.0,20]]))>0.5)    # Not_Infected

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

# Save the model
ann.save("ann_model.keras")


# Save the scaler
with open("ann_scaler.pkl", "wb") as file:
    pickle.dump(sc, file)