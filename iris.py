import ctypes
from sklearn import datasets,model_selection
import pandas as pd
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudart64_100.dll")
hllDll1 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cublas64_100.dll")
from tensorflow import keras


iris=datasets.load_iris()
X=iris.data
y=iris.target
y=keras.utils.to_categorical(y)
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.4, random_state=11)


print(X[:20,:])

model=keras.Sequential()
model.add(keras.layers.Input((4,)))
model.add(keras.layers.Dense(300,'relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(300,'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(300,'relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(300,'relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(3,'softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=150,validation_split=0.4)

#model=keras.models.load_model('weights.299-0.80.hdf5')
score=model.evaluate(X_test,y_test)
print(score)
