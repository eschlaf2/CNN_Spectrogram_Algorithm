import sys                                      # + path to fastai_v1 in root repo directory.
sys.path.insert(1, '../')

import numpy as np                              # import all required packages.
import scipy
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from fastai_v1.imports import *
from fastai_v1.transforms import *
from fastai_v1.conv_learner import *
from fastai_v1.model import *
from fastai_v1.dataset import *
from fastai_v1.sgdr import *
from fastai_v1.plots import *
from sklearn import metrics

def hannspecgramc(data,time):                  # Given data, compute spectrogram images.
    flo = 30                                   # see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5290731/
    fhi = 250
    movingwin    = [0.200,0.005]               # Spectrogram interval is 0.2 s, with stepsize 0.005 s.
    Fs = 1/(time[1]-time[0])                   # Sampling frequency [Hz]
    Fs = Fs[0];
    iStart = 1;
    iWin   = int(np.round(movingwin[0]*Fs/2)*2);
    iStep  = int(np.round(movingwin[1]*Fs/2)*2);
    T      = iWin/Fs;
    df     = 1/T;
    fNQ    = Fs/2;
    freq   = range(0,int(iWin/2+1))*df;
    counter=1;
    S      = np.zeros([int(np.ceil(len(time)/iStep)), int(iWin/2+1)]);
    times  = np.zeros([int(np.ceil(len(time)/iStep)), 1]);
    while iStart+iWin < len(time):                  # For each interval
        dnow    = data[iStart:iStart+iWin-1];       # ... get data
        dnow    = dnow - np.mean(dnow);             # ... remove mean
        dnow    = np.hanning(len(dnow))*np.squeeze(dnow);  # ... Hanning taper
        spectrum= np.real(scipy.fft.fft(dnow)*np.conjugate(scipy.fft.fft(dnow)));  # ... compute power
        S[counter,:]   = spectrum[:int(iWin/2+1)];  # ... and save it,
        times[counter] = time[int(iStart+iWin/2)];  # ... and save center time of interval.

        counter=counter+1;
        iStart = iStart + iStep;
    S = S[:counter-1,:];
    times = times[:counter-1];
    S = S[:,(freq>=flo)&(freq<=fhi)]                # Return spectrum in frequency range [flo,fhi]
    freq = freq[(freq>=flo)&(freq<=fhi)]            # ... and frequency axis in same range.
    return S,times,freq

def make_spectra_image_files(data, time):           # Given data, compute spectrogram image for each 1 s interval.
    
    start_time_dict = {}; stop_time_dict = {}
    Fs = 1/(time[1]-time[0])                        # Sampling frequency [Hz]
    Fs = Fs[0]; 
    winSize = int(np.round(Fs));                    # Divide data into 1 s intervals,
    winStep = int(np.round(Fs/2))                   # ... with 0.5 s stepsize
    
    i_stop = 0; i=0; counter = 0;
    while time[i_stop] < time[-1]-1:                # For times to end of data (can change to any value),
        i_start = i
        i_stop  = i_start + winSize - 1

        dspec = data[i_start:i_stop]                # Get data in interval.
        dspec = dspec - np.mean(dspec)
        t     = time[i_start:i_stop]
        S0,S_times,S_freq = hannspecgramc(dspec,t)  # Compute spectrogram for this interval.
        
        t_smooth = 11                               # Smooth the spectra.
        dt_S     = S_times[2]-S_times[1]
        S_smooth = gaussian_filter(S0,1)

        A = np.log10(S_smooth)                      # Scale it.
        A = np.flipud(A)

        name = 'demo_spectra_images/test/img_'+ str(counter)+'.jpg'
        plt.imsave(name,A,cmap='jet')               # Save it as image.

        new_name = str(counter)+'.jpg'
        start_time_dict[new_name] = time[i_start]
        stop_time_dict[new_name]  = time[i_stop]

        i = i_start + winStep;
        counter=counter+1;
        
    return start_time_dict, stop_time_dict

def compute_CNN(PATH, start_time_dict, stop_time_dict):
    # where the data is
    # PATH = "demo_data"
    # using resnet architecture
    arch = resnet34
    # size of square image in pixels
    sz = 44
    # transforms used on training data
    transforms_up_down = [RandomScale(sz,1.2),RandomRotate(1)]
    tfms = tfms_from_model(arch,sz,crop_type = CropType.NO,aug_tfms=transforms_up_down)
    # data: comes from PATH, used tfms on training data, bs of 8 for training data, test data located in test folder
    data = ImageClassifierData.from_paths(PATH,tfms=tfms,bs=8,test_name='test')
    # load in pretraine
    state = torch.load('../full_trained_model.pkl',map_location=torch.device('cpu')) # remove map_location parameter if on GPU
    learn2 = ConvLearner.pretrained(arch,data,precompute=False)
    learn2.model.load_state_dict(state)
    
    log_preds_test = learn2.predict(is_test=True)
    preds_test = np.argmax(log_preds_test,axis=1)
    probs_test = np.exp(log_preds_test[:,1])
    
    test_names = np.empty_like(data.test_ds.fnames)
    for i in range(len(data.test_ds.fnames)):
        test_names[i] = data.test_ds.fnames[i][9:]
        test_df = pd.DataFrame(data = test_names,columns = ['image_number'])
        test_df['prediction'] = preds_test
        test_df['probability'] = probs_test
        test_df['start time'] = test_df['image_number'].map(start_time_dict)
        test_df['stop time']  = test_df['image_number'].map(stop_time_dict)
        
    return test_df