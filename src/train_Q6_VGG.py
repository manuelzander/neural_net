'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
import os

from src.utils.data_utils import get_FER2013_data
from keras import backend as K
import pickle

#import matplotlib.pyplot as plt


K.set_image_data_format('channels_first')

batch_size = 32
num_classes = 7
epochs = 20
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'VGG_07_16_30.h5'

num_training = 25709
num_validation = 0
num_test = 3589

print("LOAD DATA")
data = get_FER2013_data(num_training, num_validation, num_test, subtract_mean=True)

x_train = data['X_train']
y_train = data['y_train']

# This is effectively our validation set
# In the model this is called X_test and y_test though
x_test = data['X_test']
y_test = data['y_test']

# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = get_FER2013_data(num_training, num_validation, num_test)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(1,48,48)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#history = None

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    filepath = 'saved_models/Model_Final{epoch:d}' + '.h5'
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, mode='auto')
    
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4,
                        callbacks=[checkpointer])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
model_path = os.path.join(save_dir, model_name)



model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

print (history.history.keys())


with open("pickled_history", 'wb') as handle:
    pickle.dump(history.history, handle)

'''
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
