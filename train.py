# Python file to train the model. 
# Takes in one argument, the name the user wants to save
# the trained model as.

##############
test = False #
##############

import os
import sys
import math
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
from keras.preprocessing.image import ImageDataGenerator
from keras import *
from keras.applications import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import matplotlib.pyplot as plt

# Set seed
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Set main directory
base_dir = os.getcwd()

# Set global variables
group1 = input("Folder name of first group (Case Sensitive): ")
print(str(len(os.listdir(group1))) + " files found in folder")
group2 = input("Folder name of second group (Case Sensitive): ")
print(str(len(os.listdir(group2))) + " files found in folder")
num_of_cat1_images = len(os.listdir(group1))
num_of_cat2_images = len(os.listdir(group2))
batch = int(input("Select batch size: "))
epoches = int(input("Select number of epoches: "))
while (True):
	model_to_use = int(input("Select models (1: VGG16, 2: VGG19, 3: InceptionResNetV2, 4: InceptionV3, 5: NASNetLarge): "))
	if ((model_to_use == 1) or (model_to_use == 2) 
		or (model_to_use == 3) or (model_to_use == 4) or (model_to_use == 5)):
		break;
	print("Invalid Input. Please choose a number between 1 and 5.")
labels = [group1, group2]

train_dir = os.path.join(base_dir, 'train_' + group1 + "_" + group2)
train_num = len(os.listdir(os.path.join(train_dir,group1))) + len(os.listdir(os.path.join(train_dir,group2)))
validation_dir = os.path.join(base_dir, 'validate_' + group1 + "_" + group2)
validation_num = len(os.listdir(os.path.join(validation_dir,group1))) + len(os.listdir(os.path.join(validation_dir,group2)))
test_dir = os.path.join(base_dir, 'test_' + group1 + "_" + group2)
test_num = len(os.listdir(os.path.join(test_dir,group1))) + len(os.listdir(os.path.join(test_dir,group2)))


save_name = 'VGG16'
if model_to_use == 2 :
	save_name = 'VGG19'
elif model_to_use == 3 :
	save_name = 'InceptionResNetV2'
elif model_to_use == 4 :
	save_name = 'InceptionV3'
elif model_to_use == 5 :
	save_name = 'NASNetLarge'
	
if not (os.path.isdir(train_dir) or os.path.isdir(validation_dir) or os.path.isdir(test_dir)):
	print("One or more of the train/validation/test directory does not exist, have you ran the script to create them?")
	sys.exit()
	
# Prevent image too large error
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Surpress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def main():

	# Create the data_generators for the train and validation data
	if (model_to_use != 5): 
		size = 240
	else :
		size = 331	
	train_generator = create_ImageDataGenerator(train_dir,size)
	validation_generator = create_ImageDataGenerator(validation_dir,size)
	
	# Create the transfer learning model
	model = create_transfer_learning_model(model_to_use)
	
	# Train model, takes in four inputs, and saves the best model
	history = train(model, train_generator, validation_generator, save_name)
	
	# Plot graphs of the training progression
	plot_graph(history)
	
	# Test the model
	test_acc = test_model(save_name, size)
	
	# Get validation accuracy of iteration with lowest validation loss
	dict = {}
	for index in range(len(history.history['val_loss'])):
		dict[history.history['val_loss'][index]] = history.history['val_acc'][index]
	
	history.history['val_loss'].sort()
	val_acc = dict[history.history['val_loss'][0]]
	
	# Print the in-sample and out-of-sample accuracy
	print("In-sample accuracy: " + str(val_acc))
	print("Out-of-sample accuracy: " + str(test_acc))

	
	
def test_model(save_name, size):
	model = models.load_model(os.path.join(train_dir,save_name))
	test_datagen = ImageDataGenerator(rescale=1./255)
	test_generator = test_datagen.flow_from_directory(
			test_dir,
			target_size=(size, size),
			batch_size=batch,
			class_mode='categorical')

	test_loss, test_acc = model.evaluate_generator(test_generator, steps=math.ceil(test_num/batch))

	return test_acc
	
def plot_graph(history):

	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(acc))

	plt.plot(epochs, acc, 'b', label='Training acc')
	plt.plot(epochs, val_acc, 'r', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()

	plt.show()
	
def train(model,train_generator, validation_generator,save_name):
	
	checkpoint = ModelCheckpoint(os.path.join(train_dir,save_name),monitor = 'val_loss', save_best_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min')
	
	history = model.fit_generator(
      train_generator,
      steps_per_epoch=math.ceil(train_num/batch),
      epochs=epoches, 
	  callbacks = [checkpoint,reduce_lr],
      validation_data=validation_generator,
      validation_steps=math.ceil(validation_num/batch))
	
	return history
	
def create_transfer_learning_model(model_to_use):

	base_model = VGG16(weights='imagenet', include_top=False)
	if model_to_use == 2 :
		base_model = VGG19(weights='imagenet', include_top=False)
	elif model_to_use == 3 :
		base_model = InceptionResNetV2(weights='imagenet', include_top=False)
	elif model_to_use == 4 :
		base_model = InceptionV3(weights='imagenet', include_top=False)
	elif model_to_use == 5 :
		base_model = NASNetLarge(weights='imagenet', include_top=False)
	
	# Create simple custom layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.5)(x)
	predictions = Dense(2, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)

	# Lock the weights of the hidden layers
	for layer in base_model.layers:
		layer.trainable = False

	opt = Adam() # Use Adam as the optimizer
	model.compile(optimizer=opt,
				  loss='binary_crossentropy',
				  metrics=['accuracy'])
	
	return model
	
	
	
def create_ImageDataGenerator(dir, target_size):
	# All images will be rescaled by 1./255
	datagen = ImageDataGenerator(rotation_range=40,
			width_shift_range=0.3,
			height_shift_range=0.3,
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			vertical_flip=True,
			fill_mode='nearest')

	generator = datagen.flow_from_directory(
			dir, # This is the input, target directory
			target_size=(target_size, target_size), # All images will be resized to target_size x target_size
			batch_size=batch,
			class_mode='categorical')
			
	return generator


	
	
	
	
	
	

if __name__ == '__main__':
	if not (test):
		main()