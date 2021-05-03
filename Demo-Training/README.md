# Example training of the CNN spike ripple detector

The CNN spike ripple detector: a method to classify spectrograms from EEG data using a convolutional neural network (CNN).

----

## Demo

To run a demonstration of **training** the CNN spike ripple detector,

- Prepare the [environment](../README.md/#environment)
- Open and run the notebook [Demo_Train_CNN.ipynb](Demo_Train_CNN.ipynb)

This demonstration trains the model on data from `train` and `valid` folders, tests on data from `test` folder.

## Data Structure

To run, you must have a data folder of the following structure:

data/

├── train/

    ├── Yes
	
    ├── No
	
├── valid/

    ├── Yes
	
    ├── No
	
├── test/

The `Yes` and `No` subfolders contain positive and negative case images on which we train the model. The `test` folder contains uncategorized images on which we test the model.
