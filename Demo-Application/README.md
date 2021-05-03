# Example application of the CNN spike ripple detector

The CNN spike ripple detector: a method to classify spectrograms from EEG data using a convolutional neural network (CNN).

----

## Demo

To run a demonstration of the CNN spike ripple detector,

- Prepare the [environment](../README.md/#environment)
- Open and run the notebook [Demo_Apply_CNN.ipynb](Demo_Apply_CNN.ipynb)

This demonstration detects spike ripples in the files (see [demo_data](./demo_data))

`data.csv` : the voltage time series

`time.csv` : the time axis (units seconds)

These data were simulated using [this code](https://github.com/Mark-Kramer/Spike-Ripple-Detector-Method/tree/simulations/Simulations) from [this paper](https://pubmed.ncbi.nlm.nih.gov/27988323/).

For application, the `test` folder contains new test data to be evaluated by the pretrained model (`full_trained_model.pkl`). For the code to run with this library, the `Yes` and `No` subfolders of `train` and `valid` cannot be empty: fill them with a few images from your test data -- this will not affect the output.
