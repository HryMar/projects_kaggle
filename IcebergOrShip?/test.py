from keras.layers import Conv2D, MaxPooling2D,Dense,Reshape
from keras.models import Model
from keras.layers import BatchNormalization, Input
from keras import optimizers
import numpy as np
import json
from time import time
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.optimizers import SGD

train_file = open('train.json')
train_data = json.load(train_file)


def data_formatting(train_data, num_classes=2):
    images = []
    labels = []
    for train_sample in train_data:
        images.append(train_sample['band_1'])
        labels.append(np.eye(num_classes)[train_sample['is_iceberg']])
    images = np.array(images).reshape((len(train_data), 75, 75, 1))
    labels = np.array(labels).reshape((len(train_data), 1, 2))
    return images, labels

def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)

def Ice(optimizer=None, input_width=75, input_height=75, nChannels=1):

    inputs = Input((input_height, input_width, nChannels))
    conv1 = Conv2D(1, (20, 20), activation='relu', border_mode='valid', init='glorot_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1,1))(conv1)
    bn1 = BN()(pool1)

    conv2 = Conv2D(1, (20, 20), activation='relu', border_mode='valid',init='glorot_normal')(bn1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1,1))(conv2)
    bn2 = BN()(pool2)

    conv3 = Conv2D(1, (15, 15), activation='relu', border_mode='valid',init='glorot_normal')(bn2)
    bn3 = BN()(conv3)

    conv4 = Conv2D(1, (10, 10), activation='relu', border_mode='valid',init='glorot_normal')(bn3)
    bn4 = BN()(conv4)

    conv5 = Conv2D(1, (10, 10), activation='relu', border_mode='valid',init='glorot_normal')(bn4)
    bn5 = BN()(conv5)
    conv5 = Reshape((-1, 9))(bn5)

    conv6=Dense(2, activation="softmax",init='glorot_normal')(conv5)

    model = Model(input=inputs, output=conv6)

    # if optimizer is not None:
    #     model.compile(loss="adam", optimizer=optimizer, metrics=['binary_crossentropy'])

    return model

data_x, data_l = data_formatting(train_data)
train_data_x, test_data_x, train_data_l, test_data_l = train_test_split(data_x, data_l, test_size=0.3, random_state=137)
print(train_data_x.shape, train_data_l.shape)

#model=load_model('weights.hdf5')
#model=load_model('meta.hdf5')

model = Ice()
model.summary()
adam = optimizers.adam(lr=0.00001)
model.compile(optimizer=SGD(lr=0.01,momentum=0.9),loss="binary_crossentropy",  metrics=["accuracy"])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
checkpointer = ModelCheckpoint(filepath='weights.hdf5',monitor='val_loss', verbose=1, save_best_only=True)

model.fit(x=train_data_x,y=train_data_l,validation_data=(test_data_x, test_data_l), batch_size=512,epochs=500,callbacks=[tensorboard,checkpointer])
print(model.evaluate(test_data_x, test_data_l, batch_size=512))

model.save("meta.hdf5")
