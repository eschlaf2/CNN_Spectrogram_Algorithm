import sys
sys.path.insert(1, '../')
import matplotlib.pyplot as plt
from fastai_v1.imports import *
from fastai_v1.transforms import *
from fastai_v1.conv_learner import *
from fastai_v1.model import *
from fastai_v1.dataset import *
from fastai_v1.sgdr import *
from fastai_v1.plots import *

def check_image_count(IMAGE_PATH):

    # avoid having a minibatch of size 1 (normalization issues later on)
    train_yes_path = IMAGE_PATH+"/train/Yes"
    train_no_path = IMAGE_PATH+"/train/No"
    valid_yes_path = IMAGE_PATH+"/valid/Yes"
    valid_no_path = IMAGE_PATH+"/valid/No"
    if len([name for name in os.listdir(train_yes_path) if ".jpg" in name])%8 == 1:
        rname = train_yes_path+"/"+os.listdir(train_yes_path)[0]
        os.remove(rname)
    if len([name for name in os.listdir(train_no_path) if ".jpg" in name])%8 == 1:
        rname = train_no_path+"/"+os.listdir(train_no_path)[0]
        os.remove(rname)
    if len([name for name in os.listdir(valid_yes_path) if ".jpg" in name])%8 == 1:
        rname = valid_yes_path+"/"+os.listdir(valid_yes_path)[0]
        os.remove(rname)
    if len([name for name in os.listdir(valid_no_path) if ".jpg" in name])%8 == 1:
        rname = valid_no_path+"/"+os.listdir(valid_no_path)[0]
        os.remove(rname)

class RandomFlipUD(CoordTransform):
    # make a transform to flip images along vertical axis, creating artificial expansion of training dataset
    def __init__(self, tfm_y=TfmType.NO, p=0.5):
        super().__init__(tfm_y=tfm_y)
        self.p=p
    def set_state(self): self.store.do_flip = random.random()<self.p
    def do_transform(self, x, is_y): return np.flipud(x).copy() if self.store.do_flip else x

def get_the_data(NEWPATH, arch):
    # using resnet architecture
    # arch = resnet34
    # define transorms of the data
    sz = 44
    # flip horizontally to artificially create more trianing data
    transforms_up_down = [RandomFlipUD(),RandomScale(sz,1.2),RandomRotate(1)]
    # make square without cropping (skew down)
    tfms = tfms_from_model(arch,sz,crop_type = CropType.NO,aug_tfms=transforms_up_down)
    # get data from path with transforms, batch size 8, test data in 'test' folder
    data = ImageClassifierData.from_paths(NEWPATH,tfms=tfms,bs=8,test_name='test')
    return data

class EarlyStopping(Callback):
    # stop training early if validation loss does not improve after patience = 5 iterations
    # load best model
    def __init__(self, learner, save_path, enc_path=None, patience=5):
        super().__init__()
        self.learner=learner
        self.save_path=save_path
        self.enc_path=enc_path
        self.patience=patience
    def on_train_begin(self):
        self.best_val_loss=100
        self.num_epochs_no_improvement=0
    def on_epoch_end(self, metrics):
        val_loss = metrics[0]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_epochs_no_improvement = 0
            self.learner.save(self.save_path)
            if self.enc_path is not None:
                self.learner.save_encoder(self.enc_path)
        else:
            self.num_epochs_no_improvement += 1
        if self.num_epochs_no_improvement > self.patience:
            print(f'Stopping - no improvement after {self.patience+1} epochs')
            return True
    def on_train_end(self):
        print(f'Loading best model from {self.save_path}')
        self.learner.load(self.save_path)
        
def train_the_model(data,arch):
   
    #train the model
    learn = ConvLearner.pretrained(arch,data,precompute=True)
    lr = 1e-2
    learn.fit(lr,1)
    learn.precompute = False
    learn.fit(1e-3,3,cycle_len=1)
    learn.unfreeze()
    lr = np.array([1e-4,1e-3,1e-2])
    cb = [EarlyStopping(learn,save_path='best_mod',patience = 6)]
    learn.fit(lr,6,cycle_len=1,cycle_mult=2,callbacks=cb)
    torch.save(learn.model.state_dict(),'demo_training_saved_model.pkl')

def test_the_model(data,arch):
    # load in pretrained
    filename = 'demo_training_saved_model.pkl';
    print('Using pretrained model '+filename)
    state = torch.load(filename,map_location=torch.device('cpu')) # remove map_location parameter if on GPU
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

    return test_df