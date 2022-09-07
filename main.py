import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import datetime
from argparse import ArgumentParser

from model import DCNN
from dataset import prepare_dataloader
from train import train_model, test_model
from utils.dataset_helper import gen_datset_split_file
from utils.model_helper import load_pretrained_weights

def argument_parse():
    parser = ArgumentParser()
    parser.add_argument('--lr', type = float, default=1e-4, help = "Learning Rate.")  
    parser.add_argument('--momentum', type = float, default=0.9, help = "Momentum.")  
    parser.add_argument('--wt_decay', type = float, default=0.0005, help = "Weight Decay.")  

    parser.add_argument('--batch_size', type = int, default=20, help = "Batch Size for training.")
    parser.add_argument('--shuffle', type = bool, default=True, help = "Shuffle train dataset.")
    parser.add_argument('--dropLast', type = bool, default=True, help = "Drop last incomplete dataset batch.")
    parser.add_argument('--numWorkers', type = int, default=4, help = "Number of worker threads for training.")
    parser.add_argument('--epochs', type = int, default=350, help = "Number of epochs during training.")

    parser.add_argument('--dataset_path', type = str, default= "../dataset/Tobacco3482", help = "Dataset storage loaction relative path.")  
    parser.add_argument('--gen_ds_split_ids', type = bool, default=True, help = "Generate new dataset split ids.")
    parser.add_argument('--train_per_cls', type = int, default=100, help = "Number of training images per class.")
    parser.add_argument('--valSplit', type = float, default=0.2, help = "Validation split percentage.")

    parser.add_argument('--img_h', type = int, default=227, help = "Input Image height")
    parser.add_argument('--img_w', type = int, default=227, help = "Input Image width")

    parser.add_argument('--dropout', type = float, default=0.2, help = "Dropout probability.")
    parser.add_argument('--pretrained_wt_path', type = str, default= "/content/drive/MyDrive/Kaggle/alexnet_weights.h5", help = "Dataset storage loaction relative path.")  #../alexnet/pytorch/alexnet.pth,   ../alexnet/alexnet-owt-4df8aa71.pth
    parser.add_argument('--model_ckpt_path', type = str, default= "/content/checkpoint/ckpt.pth", help = "Dataset storage loaction relative path.")  
    parser.add_argument('--save_ckpt_after', type = int, default=1, help = "Save model checkpoint after n epochs.")
    parser.add_argument('--test', type = bool, default=False, help = "Test mode.")
    parser.add_argument('--log_path', type = str, default= "/content/logs/", help = "Dataset storage loaction relative path.")  
    parser.add_argument('--find_mean_std', type = bool, default=False, help = "Calculate mean and std of dataset?")
    args = parser.parse_args()
    return args

    
if __name__=="__main__":
    args = argument_parse()

    #Set up logging folder as per current data and time.
    now = datetime.datetime.now()
    dt = now.strftime("%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(log_dir = os.path.join(args.log_path,dt))

    #Set up cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    #Generate train, val, test files whcih contains dataset spliting ids.
    if args.gen_ds_split_ids:
        print(f'Generating dataset files...')
        gen_datset_split_file(args)

    train_dl, val_dl, test_dl = prepare_dataloader(args)
    dcnn_model = DCNN(args).to(device)
    optimizer = optim.SGD(dcnn_model.parameters(), args.lr, momentum = args.momentum, weight_decay=args.wt_decay)
    loss_fn = nn.CrossEntropyLoss()

    if args.test:
        print(f'Testing....')
        test_model(args, dcnn_model, test_dl, loss_fn, writer, device)   
    else:
        #load pretraine weights
        if args.pretrained_wt_path!= '':
            state_dict = load_pretrained_weights(args)
            dcnn_model.load_state_dict(state_dict, strict = False) 
            dcnn_model.eval()
        train_model(args, train_dl, val_dl, dcnn_model, optimizer, loss_fn, writer, device)
        