from vgg_imp import VGG16_model
from keras.optimizers import Adam
opt = Adam(lr=0.001)
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

model = VGG16_model()
model.summary()

model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])


trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="D:\\train\\",target_size=(224,224))
#tsdata = ImageDataGenerator()
#testdata = tsdata.flow_from_directory(directory="D:\\train\\", target_size=(224,224))

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=5,generator=traindata,epochs=5)




