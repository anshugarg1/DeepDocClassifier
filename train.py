import imp
import torch
import numpy as np
from torchmetrics  import Accuracy
from utils.dataset_helper import create_label_dict
from utils.model_helper import save_checkpoint, conf_mat, plot_curve, save_lists, read_lists
    
def train_model(args, train_dl, val_dl, dcnn_model, optimizer, loss_fn, writer, device):
    accuracy = Accuracy().to(device)

    #For manually plotting loss and accuracy curves
    # tr_loss_ls = []
    # val_loss_ls = []
    # tr_acc_ls = []
    # val_acc_ls = []

    for epoch in np.arange(args.epochs):
        print(f'..Epoch..: {epoch}')
        
        #Training
        train_avg_epoch_loss = 0
        train_avg_epoch_acc = 0
        dcnn_model.train(True)

        for train_count, (sample, target) in enumerate(train_dl):
            sample = sample.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = dcnn_model(sample)

            acc = accuracy(pred, target)
            train_avg_epoch_acc += acc
            
            loss = loss_fn(pred, target)
            train_avg_epoch_loss += loss
            loss.backward()
            optimizer.step()
            
            if train_count%10==0:
                complete = (train_count*100)/len(train_dl) 
                print(f'Epoch: {epoch}.... {complete}% completed.... Training Loss: {loss}.... Accuracy: {(acc*100)}')
        
        #Validation
        val_avg_epoch_loss = 0
        val_avg_epoch_acc = 0
        dcnn_model.train(False)
        for val_count, (sample, target) in enumerate(val_dl):
            sample = sample.to(device)
            target = target.to(device)

            pred = dcnn_model(sample)
            loss = loss_fn(pred, target)
            val_avg_epoch_loss += loss

            acc = accuracy(pred, target)
            val_avg_epoch_acc += acc

            if val_count%10==0:
                complete = (val_count*100)/len(val_dl) 
                print(f'Epoch: {epoch}.... {complete}% completed.... Validation Loss: {loss}.... Accuracy: {acc*100}')

        #For manually plotting loss and accuracy curves
        # tr_loss_ls.append((train_avg_epoch_loss/(train_count+1)).data.cpu().numpy())
        # val_loss_ls.append((val_avg_epoch_loss/(val_count+1)).data.cpu().numpy())
        # tr_acc_ls.append(((train_avg_epoch_acc*100)/(train_count+1)).data.cpu().numpy())
        # val_acc_ls.append(((val_avg_epoch_acc*100)/(val_count+1)).data.cpu().numpy())

        #Logging Average Loss and Accuracy values for Training and Validation per epoch
        writer.add_scalar('Training Loss', train_avg_epoch_loss/(train_count+1), epoch)
        writer.add_scalar('Validation Loss', val_avg_epoch_loss/(val_count+1), epoch)
        writer.add_scalar('Training Accuracy', (train_avg_epoch_acc*100)/(train_count+1), epoch)
        writer.add_scalar('Validation Accuracy', (val_avg_epoch_acc*100)/(val_count+1), epoch)

        print(f'Epoch: {epoch}.... Avg Training Loss: {train_avg_epoch_loss/(train_count+1)}.... Avg. Training Accuracy: {(train_avg_epoch_acc*100)/(train_count+1)}')
        print(f'Epoch: {epoch}.... Avg Validation Loss: {val_avg_epoch_loss/(val_count+1)}.... Avg. Validation Accuracy: {(val_avg_epoch_acc*100)/(val_count+1)}')

        # save checkpoint 
        temp = epoch%args.save_ckpt_after
        print(f'Save checkpoint {temp}')
        if epoch%args.save_ckpt_after == 0:
            save_checkpoint(args, dcnn_model)
    
    #For manually plotting loss and accuracy curves
    # save_lists(tr_loss_ls, 'results/train_loss.txt')
    # save_lists(val_loss_ls, 'results/val_loss.txt')
    # save_lists(tr_acc_ls, 'results/train_acc.txt')
    # save_lists(val_acc_ls, 'results/val_acc.txt')

    #For manually plotting loss and accuracy curves
    # read_lists('results/train_loss.txt')
    # read_lists('results/val_loss.txt')
    # read_lists('results/train_acc.txt')
    # read_lists('results/val_acc.txt')

    #For manually plotting loss and accuracy curves
    # plot_curve(tr_loss_ls, val_loss_ls, 'Loss')
    # plot_curve(tr_acc_ls, val_acc_ls, 'Accuracy')
    

def test_model(args, dcnn_model, test_dl, loss_fn, writer, device):
    #load model checkpoint
    ckpt = torch.load(args.model_ckpt_path)

    ## Debugging
    # for name, params in ckpt.items():
    #     print(f'key: {name} and value shape: {params.data.shape}')

    dcnn_model.load_state_dict(ckpt)
    dcnn_model.eval()

    test_avg_epoch_loss = 0
    test_avg_epoch_acc = 0
    accuracy = Accuracy()
    model_pred = []
    ground_truth = []
        
    for test_count, (sample, target) in enumerate(test_dl):
        sample = sample.to(device)
        target = target.to(device)

        pred = dcnn_model(sample)

        # print(f'Pred: {pred}')
        # print(f'GT: {target}')
        # print(f'Max Pred: {torch.max(pred, 1)}')
        # print(f'Max Pred[1]: {torch.max(pred, 1)[1]}')

        model_pred.extend((torch.max(pred, 1)[1]).cpu().numpy())
        ground_truth.extend((target).data.cpu().numpy())

        loss = loss_fn(pred, target)
        test_avg_epoch_loss += loss

        acc = accuracy(pred, target)
        test_avg_epoch_acc += acc

        if test_count%10==0:
            complete = (test_count*100)/len(test_dl) 
            print(f'{complete}% completed.... Test Loss: {loss}.... Accuracy: {acc*100}')
        
        #Logging Loss and accuracy values per batch
        writer.add_scalar('Test Loss', loss, test_count)
        writer.add_scalar('Test Accuracy', acc, test_count)    
    
    #Plot confusion matrix - present in results folder.
    classes = create_label_dict(args.dataset_path).keys()
    print(f'Classes: {classes}')
    
    conf_mat(classes, model_pred, ground_truth)
    print(f'Avg Test Loss: {test_avg_epoch_loss/(test_count+1)}.... Avg. Test Accuracy: {(test_avg_epoch_acc*100)/(test_count+1)}')
    