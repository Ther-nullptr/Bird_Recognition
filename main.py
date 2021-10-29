# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%


# %%
import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch import optim
from torchvision import transforms

from tqdm import tqdm
from PIL import Image
import librosa
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt


OS = 'WIN'


# %%
if(OS=='WIN'):
    FILE_PATH = "C:\\Users\\86181\\Datasets\\Bird"
    SAVE_PATH = "C:\\Users\\86181\\Datasets\\Bird\\array"
else:
    FILE_PATH = "/mnt/c/Users/86181/Datasets/Bird"
    SAVE_PATH = "/mnt/c/Users/86181/Datasets/Bird/array"
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
TRAIN_RATIO = 0.8
VALIDATE_RATIO = 0.2

warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# super parameters
batch_size = 50
lr = 0.002
epoch = 50
max_count = 300
full_data = False
raw_data = True


# %%
def convert_to_spectrogram(dir,wav_file):
    if(OS=='WIN'):
        filename = FILE_PATH + '\\' + dir + '\\' + wav_file
        filesavename = SAVE_PATH + '\\' + dir + '\\' + wav_file
    else:
        filename = FILE_PATH + '/' + dir + '/' + wav_file
        filesavename = SAVE_PATH + '/' + dir + '/' + wav_file
    if(raw_data):
        y, sr = librosa.load(filename,sr=None)
        # spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = - librosa.power_to_db(spectrogram, ref=np.max) # use positive value!
        # save the np file
        with open(filesavename,'wb')as f:
            np.save(filesavename,S_dB)
    else:
        with open(filesavename,'rb')as f:
            S_dB = np.load(f)
        S_dB = Image.fromarray(S_dB)
        return S_dB


# %%
def load_files():
    label = []
    wav_data = []
    # eye = np.eye(20)

    # get filenames
    for index, dir in enumerate(BIRD_LABEL.keys()):
        if(OS=='WIN'):
            wav_files = os.listdir(FILE_PATH + '\\' + dir)
        else:
            wav_files = os.listdir(FILE_PATH + '/' + dir)

        if(full_data == False):
            wav_files = wav_files[0:max_count]

        print("checking {}".format(dir))
        for wav_file in tqdm(wav_files):
            
            wav = convert_to_spectrogram(dir,wav_file)
            wav_data.append(wav)
            label.append(index)

    print('done')

    # shuffle
    state = np.random.get_state()
    np.random.shuffle(wav_data)
    np.random.set_state(state)
    np.random.shuffle(label)

    # divide
    length = len(label)
    train_index = int(length*TRAIN_RATIO)

    train_data = wav_data[0:train_index]
    train_label = label[0:train_index]
    validate_data = wav_data[train_index:]
    validate_label = label[train_index:]

    return train_data,train_label,validate_data,validate_label


# %%
class SoundDataSet(Dataset):
    def __init__(self,data_list,label_list,transform):
        self.transform = transform
        self.wav_data = data_list
        self.label = label_list

    def __getitem__(self, index):
        wav = self.wav_data[index]
        wav = self.transform(wav)
        label = self.label[index]
        return wav, label

    def __len__(self):
        return len(self.wav_data)


# %%
def get_loader():
    
    train_data,train_label,validate_data,validate_label = load_files()

    transform = transforms.Compose([
        transforms.Resize((50,50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,std=0.5)
    ])

    # build dataset
    train_dataset = SoundDataSet(train_data,train_label,transform)
    validate_dataset = SoundDataSet(validate_data,validate_label,transform)

    # build dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=1)

    return train_loader,validate_loader



# %%
# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)

        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)

        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.batchNorm3 = nn.BatchNorm2d(256)
        self.relu = nn.LeakyReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features=256, out_features=32)
        self.linear2 = nn.Linear(in_features=32, out_features=20)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        # x = self.batchNorm3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        # x = self.batchNorm4(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout1(x)

        x = x.view(x.shape[0], -1) # 将数据设置为1维
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x


# %%
def get_network():
    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters(),
                           lr = lr,
                           betas = (0.9, 0.999),
                           eps = 1e-8,
                           weight_decay = 0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=0.002)
    loss = nn.CrossEntropyLoss()
    cnn = cnn.to(device = device)
    return cnn, optimizer, loss,scheduler


# %%
# load data
x = np.arange(1, epoch + 1, 1)
train_loader, validate_loader = get_loader()
cnn, optimizer, loss,scheduler = get_network()


# %%
# train and validate

train_l_sum, train_acc_sum = 0.0, 0.0
validate_l_sum, validate_acc_sum = 0.0, 0.0
train_loss = []
train_acc = []
validate_loss = []
validate_acc = []
max_acc = 0
# train and validate
for i in range(epoch):
    n = 0
    print("epoch {}:".format(i + 1))
    scheduler.step()
    # train
    for X, Y in tqdm(train_loader):
        X, Y = X.to(device), Y.to(device)
        Y_hat = cnn(X)
        l = loss(Y_hat, Y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
        n += Y.shape[0]
    print("train: loss %.5f, train accuracy %.1f%%" %(train_l_sum / n, 100 * train_acc_sum / n))
    train_loss.append(train_l_sum / n)
    train_acc.append(100 * train_acc_sum / n)

    train_l_sum = 0.0
    train_acc_sum = 0.0
    # max_acc = 0

    # validate
    n = 0
    with torch.no_grad():  # 关闭梯度记录
        for X, Y in tqdm(validate_loader):
            X, Y = X.to(device), Y.to(device)
            Y_hat = cnn(X)
            l = loss(Y_hat, Y)
            validate_l_sum += l.item()
            validate_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
            n += Y.shape[0]
        print("validate: loss %.5f, validate accuracy %.1f%%" %
                (validate_l_sum / n, 100 * validate_acc_sum / n))
    if(100*validate_acc_sum/n>max_acc): # the network is getting better
        torch.save(cnn,'cnn.pkl')
        print("save the network parameters.")
    validate_loss.append(validate_l_sum / n)
    validate_acc.append(100 * validate_acc_sum / n)
    max_acc = max(validate_acc)

    validate_l_sum = 0.0
    validate_acc_sum = 0.0


# %%
plt.figure(1)
plt.plot(x, train_acc, label='train accuracy')
plt.plot(x, validate_acc, label='validate accuracy')
plt.legend()
plt.show()


# %%
plt.figure(2)
plt.plot(x, train_loss, label='train loss')
plt.plot(x, validate_loss, label='validate loss')
plt.legend()
plt.show()


