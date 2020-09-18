from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
#call the dataset incide load_and_process
from dataset import load_fer2013
#call preprocess_inpust incide load_and_process
from dataset import preprocess_input
#call the incide Convulition Neural Networks mini_XCEPTİON
from models.Conv_Neural_Network import mini_XCEPTION
from sklearn.model_selection import train_test_split
#AVX2 hatası için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
def nothing(x):
    pass

batch_size = 32
num_epochs = 10000
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/'

# data generator
edit_data = ImageDataGenerator(
                        featurewise_center=False, 
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
#using mini_XCEPTION to compile (You can use the exception models you want (tiny_XCEPTION ,mini_XCEPTION, big_XCEPTION)
model = mini_XCEPTION(input_shape, num_classes)
#compile the mini model
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()



log_file = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,patience=int(patience/4), verbose=1)
train_mini = base_path + '_mini_XCEPTION'
#We add all the paths to the md5 file..
model_adları = train_mini + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_adları, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
model.fit_generator(data_generator.flow(xtrain, ytrain,batch_size),
                    steps_per_epoch=len(xtrain) / batch_size,
                    epochs=num_epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=(xtest,ytest))