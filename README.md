# CNN_Spectrogram_Algorithm

The CNN spike ripple detector: a method to classify spectrograms from EEG data using a convolutional neural network (CNN).

----

## Usage

See folder [Demo-Application](./Demo-Application) for an example applciation of the trained CNN spike ripple detector to simulated EEG data.

See folder [Demo-Training](./Demo-Training) for an example of how to train the CNN spike ripple detector using simulated spectrogram images.

Code in folder `fastai` comes from fastai version 0.7 by Jeremy Howard: https://www.fast.ai/

## Data Structure

To run either demonstration, you must have a `data` folder of the following structure:

data/

├── train/

    ├── Yes
	
    ├── No
	
├── valid/

    ├── Yes
	
    ├── No
	
├── test/

For training, the `Yes` and `No` subfolders contain positive and negative case images on which we train the model. The `test` folder contains uncategorized images on which we test the model.

For application, the `test` folder contains new test data to be evaluated by the pretrained model (`full_trained_model.pkl`). For the code to run with this library, the `Yes` and `No` subfolders of `train` and `valid` cannot be empty: fill them with a few images from your test data -- this will not affect the output.

## Environment

Below is a step-by-step method to prepare an environment capable of running the notebooks:

0. Ensure you have both conda and pip installed

1. In terminal, load in a virtual environment with conda, give it a name (`environment_name`):

`conda env create -f new_enviro.yml -n environment_name`

`conda activate environment_name`

3. Open the jupyter console to run notebooks:

`jupyter notebook` 

4. When done, use `conda deactivate` to deactivate your virtual environment. To reload this environment in the future, use `conda activate environment_name`, skipping step 2.
