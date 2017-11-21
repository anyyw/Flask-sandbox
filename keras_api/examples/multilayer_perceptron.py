#All libraries related to Keras will be housed on the API
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

#curl <host>/v1/sequentialmodel POST <model args>
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.

#curl <host>/v1/sequentialmodel PATCH <Dense/Dropout(INT, (activation="STRING"), INT)>
#Ideally, we want to roll these up in the model creation step, or at least allow specification
#of a list of stuff to add during model creation
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#Will need to make another API for optimizers for model compilation, possible include
#the optimizer as part of the API if it isn't general purpose (applied to other models)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile should be run after every PATCH or POST call, since we want this to be stateless
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#Fitting should either be included with the above compile call or be part of the prediction API
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)