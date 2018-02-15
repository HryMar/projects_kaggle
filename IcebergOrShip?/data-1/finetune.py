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
from keras.applications.resnet50 import ResNet50

# модель = ResNet50 без голови з одним dense шаром для класифікації об'єктів на nb_classes
def get_model(cls=2):
    feature_extractor = ResNet50(include_top=False, weights='imagenet', input_shape=(75, 75, 1))
    flat = Flatten()(feature_extractor.output)
    # можна додати кілька dense шарів:
    d = Dense(nb_classes*2, activation='relu')(flat)
    # d = Dense(nb_classes, activation='softmax')(d)
    d = Dense(cls, activation='softmax')(d)
    m = Model(inputs=feature_extractor.input, outputs=d)

    # базові ознаки згорткових шарів перших рівнів досить універсальні, тому ми не будемо міняти їх ваги
    # "заморозимо" всі шари ResNet50, крім кількох останніх
    # кількість шарів, які ми "заморожуємо" - це гіперпараметр
    for layer in m.layers[:-8]:
        layer.trainable = False

    # для finetuning ми використаємо звичайний SGD з малою швидкістю навчання та моментом
    m.compile(
        optimizer=SGD(lr=0.01, momentum=0.9),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    m.summary()
    return m

# кількість класів, підставте ваше значення
model = get_model()

# for layer in model.layers[:-14]:
#         layer.trainable = False

data_x, data_l = data_formatting(train_data)
train_data_x, test_data_x, train_data_l, test_data_l = train_test_split(data_x, data_l, test_size=0.3,
                                                                        random_state=137)

# model=load_model('weights.hdf5')
# model=load_model('meta.hdf5')

model = Ice()
model.summary()
adam = optimizers.adam(lr=0.00001)
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss="binary_crossentropy", metrics=["accuracy"])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
checkpointer = ModelCheckpoint(filepath='weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

model.fit(x=train_data_x, y=train_data_l, validation_data=(test_data_x, test_data_l), batch_size=512,
          epochs=500, callbacks=[tensorboard, checkpointer])
print(model.evaluate(test_data_x, test_data_l, batch_size=512))

model.save("finetune.hdf5")