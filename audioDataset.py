from torch.utils.data import Dataset
import os
import torchaudio

class AudioDataset():
    def __init__(self, dir):
        self.dir = dir
        self.file_names = [f for f in os.listdir(dir) if f.endswith('.wav')]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.file_names[idx])
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate