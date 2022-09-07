# from ast import arg
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transform import transformImg
from utils.dataset_helper import create_label_dict
from definitions import train_id_path, val_id_path, test_id_path, train, val, test


def load_split_ids(mode,ids_file):
    #Load train, val, test ids for loading data in Datasets.
    ls = []
    with open(ids_file, 'r') as f:
        for id in f:
            ls.append(id[:-1])
        # print(f'{mode} list len: {len(ls)}')
    return ls

class CustomDataset(Dataset):
    def __init__(self, args, mode, datset_id_path, transform=None):
        self.args = args
        self.transform = transform
        self.mode = mode
        self.ls = load_split_ids(mode, datset_id_path)
        self.cls_lbl_dict = create_label_dict(self.args.dataset_path)
    
    def __len__(self):
        return len(self.ls)

    def __getitem__(self, idx):
        img_path = os.path.join(os.path.realpath(self.args.dataset_path), self.ls[idx])
        cls_name  = self.ls[idx].split('/')[0]
        lbl = self.cls_lbl_dict[cls_name]
        # print(f'idx in getitem function: {idx}')
        # print(f'Path for the image file {img_path}')
        # print(f'Class name: {cls_name}')
        # print(f'Image Target Label: {lbl}')

        sample = Image.open(img_path)
        ## print(f'PIL image mode: {sample.mode}')
        # print(f'PIL image size: {sample.size}')
        # print(f'PIL image channels: {sample.getbands()}')
        
        new_sample = sample.convert(mode='RGB')
        ## new_sample.save('new_sample.jpg')
        ## print(f'new_sample channel: {list(new_sample.getdata(band=0))[:100]}')
        
        if self.transform:
            new_sample = self.transform(new_sample)
        # print(f'Final Sample size: {new_sample.shape}')
        return new_sample, lbl


def prepare_dataloader(args):
    #Set up Datasets
    tr_tf = transformImg(args.img_h, args.img_w, train)
    train_ds = CustomDataset(args, train, train_id_path, tr_tf)

    val_tf = transformImg(args.img_h, args.img_w, val)
    val_ds = CustomDataset(args, val, val_id_path, val_tf)

    test_tf = transformImg(args.img_h, args.img_w, test)
    test_ds = CustomDataset(args, test, test_id_path, test_tf)

    print(f'Train dataset length: {len(train_ds)}')
    print(f'Val dataset length: {len(val_ds)}')
    print(f'Test dataset length: {len(test_ds)}')

    #Calculate mean and std for normalisatin transformations.
    if args.find_mean_std:
        mean_tr, std_tr = get_mean_and_std(train_ds)
        mean_val, std_val = get_mean_and_std(val_ds)
        mean_tst, std_tst = get_mean_and_std(test_ds)
        print(f'Train Mean: {mean_tr}')
        print(f'Train Std: {std_tr}')

        print(f'Val Mean: {mean_val}')
        print(f'Val Std: {std_val}')

        print(f'Test Mean: {mean_tst}')
        print(f'Test Std: {std_tst}')
    
    train_dl = DataLoader(train_ds, args.batch_size, args.shuffle, num_workers=args.numWorkers, drop_last=args.dropLast)
    val_dl = DataLoader(val_ds, args.batch_size, args.shuffle, num_workers=args.numWorkers)
    test_dl = DataLoader(test_ds, args.batch_size, args.shuffle, num_workers=args.numWorkers)

    # # Debugging
    # for idx, (sample,target) in enumerate(train_dl):
        ## if idx >= 0 and idx <= 1:
            
        ##     print("Input Iamge shape: ",sample.size)
        ##     print("Target values: ", target)
        ## elif idx > 1:
        ##     break
    return train_dl, val_dl, test_dl

def get_mean_and_std(ds):
    #Calculate mean and std for normalisatin transformations.
    channels_sum, channels_squared_sum = 0, 0
    for i, (data, _) in enumerate(ds):
        # Mean over height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[1,2])
        channels_squared_sum += torch.mean(data**2, dim=[1,2])
    
    mean = channels_sum / (i+1)

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / (i+1) - mean ** 2) ** 0.5
    return mean, std

def cal_mean(ds):
    mean = [0,0,0]
    for idx, (sample,target) in enumerate(ds):
        mean[0] = mean[0] + torch.mean(sample[0])
        mean[1] = mean[1] + torch.mean(sample[1])
        mean[2] = mean[2] + torch.mean(sample[2])
    mean[0] = mean[0]/(idx+1)
    mean[1] = mean[1]/(idx+1)
    mean[2] = mean[2]/(idx+1)
    print(f'Train Mean: {mean}')
    return mean