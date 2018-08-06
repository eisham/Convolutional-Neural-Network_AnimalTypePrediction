# Building a Convolutional Neural Network

# Part-1 --> Building the CNN

# Preprocessing has been done manually by creating seperate folders for training anf test sets for each category
# *Feature scaling is absolute compulsary and would be taken care of before training the data


from keras.models import Sequential    # two ways of initilaizing a NN, either sequence of layers or graph
from keras.layers import Convolution2D # CNN is a sequence of layers and images are 2D
from keras.layers import MaxPooling2D  # This would be used in step 2
from keras.layers import Flatten       # This would be used in step 3 where we convert pooled feature maps into the large feature vector, which would become input to large fully connected layers
from keras.layers import Dense         # Used to add fully connected layers to classic Neural Network


# Initializing the CNN
classifier =  Sequential()


# Step-1 --> Convolution
classifier.add(Convolution2D(32, (3,3),input_shape= (64,64,3), activation= 'relu'))


# Step-2 --> Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Adding a second convolutional layer for improving the model
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step-3 --> Flattening
classifier.add(Flatten())


# Step-4 --> Full connection
classifier.add(Dense(output_dim=128, activation='relu'))   # Hidden layer
classifier.add(Dense(output_dim=1, activation='sigmoid'))  # Output layer


# Compliling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
