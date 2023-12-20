from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import librosa


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

### akin to torch dataloader
class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)



def preprocess_audio(audio_file, 
                     downsample=1, 
                     secs_per_fn=1, 
                     verbose=False):
    '''
    This function takes in an mp3 file and preprocesses it for our neural operator model. 

    Args:
    downsample: int = if 2, takes every other point from original audio array, if 4 every fourth point
                        this is to save computation, whats great about operators is that any resolution of
                        audio data can be processed by the same model!
    secs_per_fn: float = how many seconds of audio in one function example? 
    verbose: bool = print info about the created audio array 
    '''
    
    sps = librosa.get_samplerate(audio_file) ### samples per second
    downsample_sps = sps // downsample
    audio_data, sr = librosa.load(audio_file, sr=downsample_sps)
    audio_array = np.array(audio_data)

    ### clip array to nearest second that is an even multple of secs_per_fn
    recording_secs= len(audio_array) / downsample_sps
    
    assert recording_secs > secs_per_fn ### need at least 1 example

    ### if recording secs is 14.9 and we want 2 secs per function, we can have 7 examples...
    num_examples = int(recording_secs // secs_per_fn)

    ## num points/samples we take from the raw audio to correspond to the number of examples allowed given the sps
    ## and secs of audio per function we want
    num_samples =  int(num_examples * (downsample_sps * secs_per_fn))

    # num_samples_nearest_sec  = int(recording_secs * downsample_sps)
    audio_array = audio_array[:num_samples]

    ### reshape array so that each example is secs_per_fn of audio
    audio_array = audio_array.reshape(num_examples,
                                      int(downsample_sps*secs_per_fn))
    if verbose:
        print(f'downsample to {downsample_sps} hz')
        print(f'created dataset shape of {audio_array.shape}')

    return audio_array, sps ### keep original sps in case we need it later


class BluesDriverDataset:
    def __init__(self, config):
        self.x_train_fp= config.data.x_train_fp
        self.y_train_fp= config.data.y_train_fp
        self.x_test_fp = config.data.x_test_fp
        self.y_test_fp = config.data.y_test_fp
        self.downsample = config.data.downsample
        self.secs_per_fn = config.data.secs_per_fn
        self.ntrain_frac = config.data.ntrain_frac
        self.ntest_frac = config.data.ntest_frac

    def get_train_data(self):
        return BluesDriverNpDataset(self.x_train_fp,
                               self.y_train_fp,
                               downsample=self.downsample,
                               secs_per_fn=self.secs_per_fn,
                               nexamples_frac=self.ntrain_frac
                               )
    
    def get_test_data(self):
        return BluesDriverNpDataset(self.x_test_fp,
                               self.y_test_fp,
                               downsample=self.downsample,
                               secs_per_fn=self.secs_per_fn,
                               nexamples_frac=self.ntest_frac,
                               )


### handle secs per function 
class BluesDriverNpDataset(Dataset):
    def __init__(self, 
                 x_file_path, 
                 y_file_path, 
                 downsample=1, 
                 secs_per_fn=1,
                 nexamples_frac=1.): ### 29 examples for 0.5 sec fns w/ inmyarmsriff
        
        x_audio_array,_ = preprocess_audio(x_file_path, 
                                           secs_per_fn=secs_per_fn,
                                           downsample=downsample,
                                           verbose=False)
        
        y_audio_array,_ = preprocess_audio(y_file_path, 
                                           secs_per_fn=secs_per_fn,
                                           downsample=downsample,
                                           verbose=False)
        
        assert nexamples_frac <= 1
        nex_x = int(x_audio_array.shape[0] * nexamples_frac)
        nex_y = int(y_audio_array.shape[0] * nexamples_frac)
        self.x = x_audio_array[:nex_x]
        self.y = y_audio_array[:nex_y] 

    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]