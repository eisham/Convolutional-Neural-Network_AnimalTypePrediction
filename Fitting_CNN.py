# Part-2 --> Fitting CNN to the images 

# Images are preprocessed - feature scaled, Augmented to compensate for less input images 
from keras.preprocessing.image import ImageDataGenerator 

# Different modifications initialized to be applied on training set
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

# Feature scaling applied for test set
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                 'dataset/training_set',
                                                 target_size=(64,64), # dimension of input image 
                                                 batch_size=32,
                                                 class_mode='binary') # This indicates how many classes we have in our dependent variable

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

# CNN model built in file 'Building_CNN' is used to fit to the processed images
classifier.fit_generator(
                         training_set,
                         steps_per_epoch=8000, # Number of images in training set
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000) # Number of images in test set