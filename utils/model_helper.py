from cProfile import label
import os
import torch
import h5py  
import itertools
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from definitions import train, val

def gen_state_dict(args):
    #Read pre-trained AlexNet weights from .h5 file to a dictionary.
    final_dict = {}
    f1 = h5py.File(args.pretrained_wt_path,'r+') 
    for k in f1.keys():
        f1[k].keys()
        if len(f1[k].keys()) != 0:
            for subKey in f1[k].keys():
                final_dict[subKey]=f1[k][subKey][()] 
    print(f'Pre-trained model state dictionary keys: {final_dict.keys()}')
    return final_dict
    
def load_pretrained_weights(args):
    #Load pre-trained AlexNet weights to DCNN model.
    new_dict = {}
    pretrianed_dict = gen_state_dict(args)
    new_dict['sq1.0.weight'] =  torch.tensor(pretrianed_dict['conv_1_W'])
    new_dict['sq1.0.bias'] =  torch.tensor(pretrianed_dict['conv_1_b'])
    
    new_dict['sq2_1.0.weight'] =  torch.tensor(pretrianed_dict['conv_2_1_W'])
    new_dict['sq2_1.0.bias'] =  torch.tensor(pretrianed_dict['conv_2_1_b'])
    new_dict['sq2_2.0.weight'] =  torch.tensor(pretrianed_dict['conv_2_2_W'])
    new_dict['sq2_2.0.bias'] =  torch.tensor(pretrianed_dict['conv_2_2_b'])
    
    new_dict['sq3.0.weight'] =  torch.tensor(pretrianed_dict['conv_3_W'])
    new_dict['sq3.0.bias'] =  torch.tensor(pretrianed_dict['conv_3_b'])
    
    new_dict['sq4_1.0.weight'] =  torch.tensor(pretrianed_dict['conv_4_1_W'])
    new_dict['sq4_1.0.bias'] =  torch.tensor(pretrianed_dict['conv_4_1_b'])
    new_dict['sq4_2.0.weight'] =  torch.tensor(pretrianed_dict['conv_4_2_W'])
    new_dict['sq4_2.0.bias'] =  torch.tensor(pretrianed_dict['conv_4_2_b'])
    
    new_dict['sq5_1.0.weight'] =  torch.tensor(pretrianed_dict['conv_5_1_W'])
    new_dict['sq5_1.0.bias'] =  torch.tensor(pretrianed_dict['conv_5_1_b'])
    new_dict['sq5_2.0.weight'] =  torch.tensor(pretrianed_dict['conv_5_2_W'])
    new_dict['sq5_2.0.bias'] =  torch.tensor(pretrianed_dict['conv_5_2_b'])
    
    new_dict['sq6.0.weight'] =  torch.transpose(torch.tensor(pretrianed_dict['dense_1_W']), 0,1)
    new_dict['sq6.0.bias'] =  torch.tensor(pretrianed_dict['dense_1_b'])
    
    new_dict['sq7.0.weight'] =  torch.transpose(torch.tensor(pretrianed_dict['dense_2_W']), 0,1)
    new_dict['sq7.0.bias'] =  torch.tensor(pretrianed_dict['dense_2_b'])
    
    return new_dict
    
def save_lists(ls, file_name):
    with open(file_name, 'w') as filehandle:
        for listitem in ls:
            filehandle.write('%s\n' % listitem)

def read_lists(file_name):
    ls = []
    # open file and read the content in a list
    with open(file_name, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            # add item to the list
            ls.append(currentPlace)
    print(ls)
    return ls

def save_checkpoint(args, model):
    torch.save(model.state_dict(), args.model_ckpt_path)

def plot_curve(tr_ls, val_ls, mode):
    print(f'Train list: {tr_ls}')
    print(f'Val list: {val_ls}')
    ls = {'train': tr_ls, 'val': val_ls}
    print(list(range(0, len(tr_ls))))

    x = list(range(0, len(tr_ls)))

    # loss_df = pd.DataFrame(ls, index=list(range(0, len(tr_ls))), columns=[train, val])
    # print(f'{mode} dataframe: {loss_df}')

    plt.figure(figsize = (12,7))
    plt.plot(x, tr_ls, label='Train'+mode)
    plt.plot(x, val_ls, label='Validation'+mode)

    # sn.lineplot(data = loss_df)
    plt.ylabel(mode)
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('results/'+mode+'.png')

def conf_mat(classes, pred, target):
    # Generate confusion matrix.
    cf_mat = confusion_matrix(target, pred)
    print(f'Confusion Matrix:\n {cf_mat}')

    #type 1
    conf_mat2 = np.zeros((10,10))
    for i in np.arange(10):
        if np.sum(cf_mat[:,i]) == 0:
            conf_mat2[:,i] = 0
        else:
            conf_mat2[:,i] = 100*cf_mat[:,i]/np.sum(cf_mat[:,i])
    df_cm = pd.DataFrame(conf_mat2, index = [i for i in classes], columns = [i for i in classes])

    #plotting
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, cmap='Blues', annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')  #, fontsize=20
    plt.title('Confusion Matrix')
    plt.savefig('results/output1.png')