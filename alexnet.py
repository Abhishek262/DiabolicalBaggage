"""
AlexNet Keras Implementation
BibTeX Citation:
@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
"""

# Import necessary packages
import argparse

# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def freeze_layer(layer,fvalue) : 
      
    if(fvalue):
        layer.trainable = False
        
    return layer

def alexnet_model(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,
	weights=None,freeze = [0,0,0,0,0,0,0,0,0]):

	# Initialize model
	alexnet = Sequential()
    

	# Layer 1
	alexnet.add(freeze_layer(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)),fvalue = freeze[1]))
    
	alexnet.add(freeze_layer(BatchNormalization(),fvalue = freeze[1]))
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2 
	alexnet.add(freeze_layer(Conv2D(256, (5, 5), padding='same'),fvalue = freeze[2] ))
	alexnet.add(freeze_layer(BatchNormalization(),fvalue = freeze[2]))
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(freeze_layer(Conv2D(512, (3, 3), padding='same'),fvalue = freeze[3]))
	alexnet.add(freeze_layer(BatchNormalization(),fvalue = freeze[3]))
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(freeze_layer((Conv2D(1024, (3, 3), padding='same')),fvalue = freeze[4]))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(freeze_layer(Conv2D(1024, (3, 3), padding='same'),fvalue = freeze[5]))
	alexnet.add(freeze_layer(BatchNormalization(),fvalue = freeze[5]))
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(freeze_layer(Dense(3072),fvalue = freeze[6]))
	alexnet.add(freeze_layer(BatchNormalization(),fvalue = freeze[6]))
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(freeze_layer(Dense(4096),fvalue = freeze[7]))
	alexnet.add(freeze_layer(BatchNormalization(),fvalue = freeze[7]))
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(freeze_layer(Dense(n_classes),fvalue = freeze[8]))
	alexnet.add(freeze_layer(BatchNormalization(),fvalue = freeze[8]))
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	alexnet.summary()
	return alexnet

def parse_args():
	"""
	Parse command line arguments.
	Parameters:
		None
	Returns:
		parser arguments
	"""
	parser = argparse.ArgumentParser(description='AlexNet model')
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	optional.add_argument('--print_model',
		dest='print_model',
		help='Print AlexNet model',
		action='store_true')
	parser._action_groups.append(optional)
	return parser.parse_args()

if __name__ == "__main__":
	# Command line parameters
	args = parse_args()

	# Create AlexNet model
	model = alexnet_model()

	# Print
	if args.print_model:
		model.summary()