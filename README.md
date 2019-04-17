# Image Classification Part 2

### Brief Summary
More tagged images were acquired from flickr. More transfer learning were trained using the new sets of tagged images. The in-sample and out-of-sample accuracy was recorded.


## Quick check on the different image tag pairs
The following models were not trained using reproducable results, which was admittedly due to my own negligence in the code. However, attempting to reproduce with the same code should still net a ±1 difference in accuracy.

## DNN
The new images can be downloaded with the same method as part one. Two new py files, split_data.py and train.py have been created and were used in this part.

The first file 'split_data.py' is used to split the data into the train, validation, and test categories. This should be done after downloading the augmenting the data.

The second file 'train.py' is used to train the data. The program will prompt the user for the input after being run. 

### Saturated vs Drab
The five different base models were selected and trained using the train.py file. The batch-size used was 32, and number of epoches 20.

- VGG16 -- test acc: 0.812, validation acc: 0.811
- VGG19 -- test acc: 0.808, validation acc: 0.808
- InceptionResNetV2 -- test acc: 0.791, validation acc: 0.833
- InceptionV3 -- test acc: 0.806, validation acc: 0.789
- NASNetLarge -- test acc: 0.833, validation acc: 0.824

### Glamorous vs Drab
The same five base models and parameters were used. **However, I do not believe the Glamorous image set should be used, as almost all of the images are of human females, which is likely to not be representative of our goals.**

- VGG16 -- test acc: 0.788, validation acc: 0.787
- VGG19 -- test acc: 0.785, validation acc: 0.784
- InceptionResNetV2 -- test acc: 0.839, validation acc: 0.854
- InceptionV3 -- test acc: 0.835, validation acc: 0.811
- NASNetLarge -- test acc: 0.871, validation acc: 0.872

### Fun vs Dull
The same five base models and parameters were used.

- VGG16 -- test acc: 0.759, validation acc: 0.790
- VGG19 -- test acc: 0.753, validation acc: 0.785
- InceptionResNetV2: 0.835, validation acc: 0.843
- InceptionV3 -- test acc: 0.825, validation acc: 0.843
- NASNetLarge -- test acc: 0.857, validation acc: 0.864

### Extra
It is interesting to note here that the Fun vs Dull model actually provides a better accuracy than the Sensational vs Drab model done in part one. To look further, I reran the models, but this time I set a seed should allow for reproducability. 

NASNetLarge (Fun-Dull) batch-size 16
- In-sample accuracy: 0.844
- Out-of-sample accuracy: 0.8616

NASNetLarge (Fun-Dull) batch-size 32
- In-sample accuracy: 0.856
- Out-of-sample accuracy: 0.849

NASNetLarge (Sensational-Drab) batch-size 16
- In-sample accuracy: 0.870
- Out-of-sample accuracy: 0.842

NASNetLarge (Sensational-Drab) batch-size 32
- In-sample accuracy: 0.855
- Out-of-sample accuracy: 0.843

## SVM
To be done...
