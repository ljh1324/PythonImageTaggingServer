import os

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import plot_model # for model description png
from keras.losses import *
from keras.activations import relu

def _add_BN_ReLU_DO(x, doPercent):
    mainlayerName = str(x.name).split('/')[0]
    x = BatchNormalization(name=mainlayerName + '_BN')(x)
    x = Activation('relu', name=mainlayerName + '_ReLU')(x)
    x = Dropout(doPercent, name=mainlayerName + '_DO')(x)
    return x

def model(imageSize, labelSize):
    '''
    branch 1 (spoint cnn branch)
    '''
    height, width = imageSize

    b1_dropout = 0.0
    m1_learningRate = 1e-3

    b1_input = Input(shape=(height, width, 3), name='B1_Input')
    #b1 = _add_BN_ReLU_DO(b1, b1_dropout)
    b1 = Conv2D(filters=16,
            kernel_size=(3, 3),
            strides=(1, 1), 
            data_format="channels_last",
            padding = 'same',
            name = 'B1C1')(b1_input)
    b1 = _add_BN_ReLU_DO(b1, b1_dropout)
    b1 = MaxPool2D(pool_size=(2, 2))(b1)
    b1 = Conv2D(filters=18,
            kernel_size=(3, 1),
            strides=(1, 1), 
            padding = 'same',
            name = 'B1C2')(b1)
    b1 = _add_BN_ReLU_DO(b1, b1_dropout)
    b1 = MaxPool2D(pool_size=(2, 2))(b1)
    b1 = Conv2D(filters=20,
            kernel_size=(3, 3),
            strides=(2, 1),
            padding = 'same',
            name = 'B1C3')(b1)
    b1 = _add_BN_ReLU_DO(b1, b1_dropout)
    b1 = Flatten(name='B1F1')(b1)
    b1 = Dense(128, name='B1D1')(b1)
    b1 = _add_BN_ReLU_DO(b1, b1_dropout)
    b1 = Dense(128, name='B1D2')(b1)
    b1 = _add_BN_ReLU_DO(b1, b1_dropout)
    b1 = Dense(128, name='B1D3')(b1)
    b1 = _add_BN_ReLU_DO(b1, b1_dropout)
    b1_output = Dense(labelSize, activation='softmax', name='softmax')(b1)

    model = Model(inputs=b1_input, outputs=b1_output)
    optimizer = Adam(lr=m1_learningRate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def loadWeight(model, dirPath):
    path = os.path.join(dirPath, model.name + '.h5')

    if os.path.exists(path):
        #print('Loading Architecture: ' + model.name)
        
        model.load_weights(path)
        print('Loading Weight : ' + model.name)

    else:
        print('Loading Weight Fail : ' + model.name)

    return model

def saveWeight(model, dirPath):
    path = os.path.join(dirPath, model.name + '.h5')
    
    print ('Saving model: ' + model.name)
    print ('Path: ' + path)
    model.save_weights(path)

def saveModelDescription(model, path, isShow):
    path = os.path.join(path, 'md_' + model.name + '.jpg')
    plot_model(model, to_file=path, show_layer_names=True, show_shapes=True)
    if isShow:
        print(model.name + ' done')

