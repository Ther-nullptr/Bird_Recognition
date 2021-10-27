import torch
from torch.utils.data import Dataset,DataLoader
import wave
import os

FILE_PATH = "/mnt/c/Users/86181/Datasets/Bird/"
BIRD_LABEL = {
    "0009": "灰雁",
    "0017": "大天鹅",
    "0034": "绿头鸭",
    "0036": "绿翅鸭",
    "0074": "灰山鹑",
    "0077": "西鹌鹑",
    "0114": "雉鸡",
    "0121": "红喉潜鸟",
    "0180": "苍鹭",
    "0202": "普通鸬鹚",
    "0235": "苍鹰",
    "0257": "欧亚鵟",
    "0265": "西方秧鸡",
    "0281": "骨顶鸡",
    "0298": "黑翅长脚鹬",
    "0300": "凤头麦鸡",
    "0364": "白腰草鹬",
    "0368": "红脚鹬",
    "0370": "林鹬",
    "1331": "麻雀"
}


class SoundDataSet(Dataset):
    def __init__(self,file):
        self.wav_data = []
        self.label = []
        for dir in BIRD_LABEL.keys():
            wav_files = os.listdir(FILE_PATH+dir)
            for wav_file in wav_files:
                self.wav_data.append(wav_file)
                self.label.append(dir)
                
        

    def __getitem__(self,index):
        wav = self.wav_data[index]
        label = self.label[index]
        return wav,label

    def __len__(self):
        return len(self.wav_data)