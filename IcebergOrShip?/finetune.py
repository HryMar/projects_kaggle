from keras.layers import Conv2D, MaxPooling2D,Dense,Reshape,Flatten,BatchNormalization, Input
from keras.models import Model,load_model
from keras.optimizers import SGD, adam
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.applications.resnet50 import ResNet50
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
from time import time
from sklearn.model_selection import train_test_split

def data_formatting(train_data, num_classes=2):
    images = []
    labels = []
    for train_sample in train_data:
        images.append(train_sample['band_1'])
        labels.append(np.eye(num_classes)[train_sample['is_iceberg']])
    images = np.array(images).reshape((len(train_data), 75, 75, 1))
    labels = np.array(labels).reshape((len(train_data), 1, 2))
    return images, labels



def get_model():
    feature_extractor = VGG16(include_top=False, weights='imagenet', input_shape=(75, 75, 3))
    R = Reshape((-1, 4*512))(feature_extractor.output)
    d1 = Dense(500, activation='relu')(R)
    d2 = Dense(50, activation='softmax')(d1)

    d = Dense(2, activation='softmax')(d2)
    m = Model(inputs=feature_extractor.input, outputs=d)


    for layer in m.layers[:-8]:
        layer.trainable = False

    m.compile(
        optimizer=SGD(lr=0.01, momentum=0.9),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    m.summary()
    return m



# for layer in model.layers[:-14]:
#         layer.trainable = False

train_file = open('train.json')
train_data = json.load(train_file)
data_x, data_l = data_formatting(train_data)
data_x = np.concatenate([data_x, data_x, data_x], axis=3)
train_data_x, test_data_x, train_data_l, test_data_l = train_test_split(data_x, data_l, test_size=0.3,
                                                                        random_state=137)

#model=load_model('fineweights.hdf5')
# model=load_model('meta.hdf5')
model = get_model()
model.summary()
adam = adam(lr=0.00001)
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss="binary_crossentropy", metrics=["accuracy"])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
checkpointer = ModelCheckpoint(filepath='fineweights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

batch_size=561

train_gen = ImageDataGenerator(
    rescale=1./75,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_gen = ImageDataGenerator(rescale=1./75)

train_generator = train_gen.flow(
    train_data_x,train_data_l,
    batch_size=batch_size)

test_generator = test_gen.flow(
    test_data_x, test_data_l,
    batch_size=batch_size)

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=2,
    epochs=1000,
    callbacks=[tensorboard, checkpointer],
    validation_data = test_generator,
    validation_steps = 2)

model.save("finetune.hdf5")
