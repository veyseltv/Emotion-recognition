#Convulition Neural Network için gerekli kütüphaneler eklenip tanımlanıyor.
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
#AVX2 hatası için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def nothing(x):
    pass

def simple_CNN(input_shape, num_classes):
    #CNN 'in tanımlanması.
    classifier = Sequential()
    # 1.1.adım Convulution Layer
    classifier.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                                 name='image_array', input_shape=input_shape))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    # 1.2.adım Pooling2D
    classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(.5))
    #2.1. adım Convulution Layer
    classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    # 2.2. adım Pooling2D
    classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(.5))

    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    #relu Layer
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    # Average pooling layer
    classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    #overfitting
    classifier.add(Dropout(.5))
    # 3.1 Convulution Layer
    classifier.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    #relu layer
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    # 3.2 Pooling2D
    classifier.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(Dropout(.5))

    classifier.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
    #GlobalAveragePooling2D does something different.
    # It applies average pooling on the spatial dimensions until each spatial dimension is one,
    # and leaves other dimensions unchanged.
    # In this case values are not kept as they are averaged.
    classifier.add(GlobalAveragePooling2D())
    classifier.add(Activation('softmax', name='predictions'))
    return classifier

#The same explanations written above are also valid for this region.
def simpler_CNN(input_shape, num_classes):

    classifier = Sequential()
    classifier.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same',name='image_array', input_shape=input_shape))

    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=16, kernel_size=(5, 5),strides=(2, 2), padding='same'))

    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(.25))

    classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'))

    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(.25))

    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(.25))

    classifier.add(Convolution2D(filters=64, kernel_size=(1, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=128, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(.25))

    classifier.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=128, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same'))

    classifier.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters=num_classes, kernel_size=(3, 3),
                                 strides=(2, 2), padding='same'))
    # Flatten
    classifier.add(Flatten())
    classifier.add(Activation('softmax', name='predictions'))
    return classifier

#The same explanations written above are also valid for this region.
def tiny_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # başlangıc ölçütleri
    img_input = Input(input_shape)
    value = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(value)

    value = BatchNormalization()(value)
    value = Activation('relu')(value)

    # module 1
    residual = Conv2D(8, (1, 1), strides=(2, 2), padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(8, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = SeparableConv2D(8, (3, 3), padding='same',kernel_regularizer=regularization, use_bias=False)(value)
    value = BatchNormalization()(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    # module 2
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(16, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = SeparableConv2D(16, (3, 3), padding='same',kernel_regularizer=regularization, use_bias=False)(value)
    value = BatchNormalization()(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    # module 3
    residual = Conv2D(32, (1, 1), strides=(2, 2),padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization,use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization,use_bias=False)(value)
    value = BatchNormalization()(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    # module 4
    residual = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(value)
    value = BatchNormalization()(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    value = Conv2D(num_classes, (3, 3),
                   padding='same')(value)
    value = GlobalAveragePooling2D()(value)
    output = Activation('softmax',name='predictions')(value)

    model = Model(img_input, output)
    return model

#The same explanations written above are also valid for this region.
def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    value = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                   use_bias=False)(img_input)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                   use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(value)
    value = BatchNormalization()(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(value)
    value = BatchNormalization()(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(value)
    value = BatchNormalization()(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(value)
    value = BatchNormalization()(value)
    value = Activation('relu')(value)
    value = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(value)
    value = BatchNormalization()(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    value = Conv2D(num_classes, (3, 3),
                   padding='same')(value)
    value = GlobalAveragePooling2D()(value)
    output = Activation('softmax',name='predictions')(value)

    model = Model(img_input, output)
    return model
#The same explanations written above are also valid for this region.
def big_XCEPTION(input_shape, num_classes):
    img_input = Input(input_shape)
    value = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    value = BatchNormalization(name='block1_conv1_bn')(value)
    value = Activation('relu', name='block1_conv1_act')(value)
    value = Conv2D(64, (3, 3), use_bias=False)(value)
    value = BatchNormalization(name='block1_conv2_bn')(value)
    value = Activation('relu', name='block1_conv2_act')(value)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(value)
    value = BatchNormalization(name='block2_sepconv1_bn')(value)
    value = Activation('relu', name='block2_sepconv2_act')(value)
    value = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(value)
    value = BatchNormalization(name='block2_sepconv2_bn')(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(value)
    residual = BatchNormalization()(residual)

    value = Activation('relu', name='block3_sepconv1_act')(value)
    value = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(value)
    value = BatchNormalization(name='block3_sepconv1_bn')(value)
    value = Activation('relu', name='block3_sepconv2_act')(value)
    value = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(value)
    value = BatchNormalization(name='block3_sepconv2_bn')(value)

    value = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(value)
    value = layers.add([value, residual])
    value = Conv2D(num_classes, (3, 3),
                   padding='same')(value)
    value = GlobalAveragePooling2D()(value)
    output = Activation('softmax',name='predictions')(value)

    model = Model(img_input, output)
    return model


if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    model = simple_CNN((48, 48, 1), num_classes)
    model.summary()
