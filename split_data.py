import os
import shutil
from PIL import Image
from PIL import ImageFile
import random

def main():

	# Prevent errors
	Image.MAX_IMAGE_PIXELS = None
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	
	split_data() # Split data into train, validate, and test
	
def split_data():
	base_dir = os.getcwd() 
	
	train_ratio = 0.7; validation_ratio = 0.15; test_ratio = 0.15
	first_folder = input("Input the file name for the first group of images: ")
	print(str(len(os.listdir(first_folder))) + " files found in folder")
	second_folder = input("Input the file name for the first group of images: ")
	print(str(len(os.listdir(second_folder))) + " files found in folder")
	labels = [first_folder, second_folder]

	train_dir = os.path.join(base_dir, 'train_' + first_folder + "_" + second_folder)
	validation_dir = os.path.join(base_dir, 'validate_' + first_folder + "_" + second_folder)
	test_dir = os.path.join(base_dir, 'test_' + first_folder + "_" + second_folder)
	
	# Create the respective directories if they do not exist
	if not os.path.exists(train_dir):
		os.mkdir(train_dir)
		print("Directory for training images created")
		
	if not os.path.exists(validation_dir):
		os.mkdir(validation_dir)
		print("Directory for validation images created")
		
	if not os.path.exists(test_dir):
		os.mkdir(test_dir)
		print("Directory for test images created")
	
	# For each category, split them into three seperate 
	for category in labels:
    
		category_size = int(len(os.listdir(os.path.join(base_dir,category))))
		
		train_size = int(train_ratio * category_size)
		validation_size = int(validation_ratio * category_size)
		test_size = category_size - (train_size + validation_size)
		category_dir = os.path.join(base_dir, category)
		
		# Copy data from category_dir to create train set for category
		train_category_dir = os.path.join(train_dir,category)
		
		if not os.path.exists(train_category_dir):
			os.mkdir(train_category_dir)
		
		fnames_all = os.listdir(category_dir)
		random.shuffle(fnames_all)
		
		fnames = fnames_all[0:train_size]
		for fname in fnames:
			src = os.path.join(category_dir, fname)
			try:
				Image.open(src)
				dst = os.path.join(train_category_dir, fname)
				shutil.copyfile(src, dst)
			except:
				continue
		print(category + "'s train directory done!")
			
		# Copy data from category_dir to create validation set for category
		validation_category_dir = os.path.join(validation_dir, category)
		
		if not os.path.exists(validation_category_dir):
			os.mkdir(validation_category_dir)
		
		fnames = fnames_all[train_size:train_size+validation_size]
		for fname in fnames:
			src = os.path.join(category_dir, fname)
			try:
				Image.open(src)
				dst = os.path.join(validation_category_dir, fname)
				shutil.copyfile(src, dst)
			except:
				continue
		print(category + "'s validation directory done!")
			
		# Copy data from category_dir to create test set for category
		test_category_dir = os.path.join(test_dir, category)
		
		if not os.path.exists(test_category_dir):
			os.mkdir(test_category_dir)   
		
		fnames = fnames_all[train_size+validation_size:]
		for fname in fnames:
			src = os.path.join(category_dir, fname)
			try: 
				Image.open(src)
				dst = os.path.join(test_category_dir, fname)
				shutil.copyfile(src, dst)
			except:
				continue
		print(category + "'s test directory done!")
        
	print("All done!")


if __name__ == '__main__':
	main()