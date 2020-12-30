# -*- coding: utf-8 -*-

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
warnings.filterwarnings("ignore")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Train_batch_size', default=256, type=int,
                    help='Train Batch size')
parser.add_argument('--Test_batch_size', default=128, type=int,
                    help='Test Batch size')
parser.add_argument('--Epoch', default=1, type=int,
                    help='Epoch')
parser.add_argument('--Lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--auto_Lr', default=True, type=bool,
                    help='if auto adjust the learning rate')
parser.add_argument('--decay', default=0.01, type=float,
                    help='the decay rate of learning rate')
parser.add_argument('--signal_lenght', default=8, type=int,
                    help='signal lenght')
parser.add_argument('--Fs', default=250, type=int,
                    help='Fs')
parser.add_argument('--window', default=0.6, type=float,
                    help='window lenght')
parser.add_argument('--step', default=0.2, type=float,
                    help='window step')
parser.add_argument('--p', default=3, type=float,
                    help='p in auc loss')
parser.add_argument('--gamma', default=0.4, type=float,
                    help='gamma in auc loss')
args = parser.parse_args()


cv = np.eye(5)

# auc loss
def auc_loss(y_pred, y_true, p=args.p, gamma=args.gamma):
    '''Soft version of AUC that uses Wilcoxon-Mann-Whitney U. statistic'''
    # Grab the logits of all the positive and negative examples
    pos = y_pred[y_true.view(-1, 1).bool()].view(1, -1)
    neg = y_pred[~y_true.view(-1, 1).bool()].view(-1, 1)
    difference = torch.zeros_like(pos * neg) + pos - neg - gamma
    masked = difference[difference < 0.0]
    return torch.div(torch.sum(torch.pow(-masked, p)), y_true.shape[0])



def slide_window(data, window=args.window ,step=args.step ,Fs=args.Fs):
    timestep = round(((data.shape[1]/Fs)-window)/step)+1 #滑动窗数量
    for i in range(0,timestep):
        if i==0:
            X = data[:,round((i*step)*Fs):round((i*step+window)*Fs)]
        else:
            b = data[:,round(i*step*Fs):round((i*step+window)*Fs)]
            X = np.hstack((X,b))

    return X.reshape(-1,timestep,int(window*Fs)),timestep




class TrainData(Dataset):
    def __init__(self, data, index, lenght=args.signal_lenght, window=args.window ,step=args.step ,Fs=args.Fs):
        self.X = np.vstack(data[index,0])
        self.X = self.X[:,0:int(lenght*Fs)]
        self.Y = np.vstack(data[index,1])

        permutation = np.random.permutation(self.X.shape[0])
        self.X = self.X[permutation,:]
        self.Y = self.Y[permutation,:]
        self.X = slide_window(self.X, window, step)[0]
        self.X = np.expand_dims(self.X,axis=1)
        
        self.len = len(self.Y)

    def __getitem__(self, index):

        return self.X[index], self.Y[index]

    def __len__(self):

        return self.len


class TestData(Dataset):
    def __init__(self, data, index, lenght=args.signal_lenght, window=args.window ,step=args.step ,Fs=args.Fs):
        self.X = np.vstack(data[index,0])
        self.X = self.X[:,0:int(lenght*Fs)]
        self.Y = np.vstack(data[index,1])

        permutation = np.random.permutation(self.X.shape[0])
        self.X = self.X[permutation,:]
        self.Y = self.Y[permutation,:]
        self.X = slide_window(self.X, window, step)[0]
        self.X = np.expand_dims(self.X,axis=1)
        
        self.len = len(self.Y)

    def __getitem__(self, index):

        return self.X[index], self.Y[index]

    def __len__(self):

        return self.len




class CNN_LSTM(nn.Module):
    def __init__(self, 
                 signal_len=args.signal_lenght, 
                 window=args.window ,
                 step=args.step ,
                 Fs=args.Fs, 
                 lstm_hidden=50):
        super(CNN_LSTM,self).__init__()
        self.linear_hidden = int(((signal_len-window)*Fs/(step*Fs)+1)*lstm_hidden)
        self.in_lstm = int((window*Fs-4)+1)

        self.kernel = [[-1,1,0,0]]
        self.kernel = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(0).cuda()
        self.kernel_weight = nn.Parameter(data=self.kernel, requires_grad=True)
        self.lstm = nn.LSTM(input_size=self.in_lstm, hidden_size=lstm_hidden)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.linear_hidden,20)
        self.linear2 = nn.Linear(20,1)
        

    def forward(self,X):

        tensor = nn.functional.conv2d(X, self.kernel_weight, stride=1, padding=0)
        tensor = torch.squeeze(tensor, 1)

        tensor, indices = torch.sort(tensor, dim=-1)

        tensor, (h_n, c_n) = self.lstm(tensor)
        tensor = self.flatten(tensor)
        tensor = self.linear1(tensor)
        tensor = nn.functional.tanh(tensor)
        tensor = self.linear2(tensor)

        out = nn.functional.sigmoid(tensor)
        
        return out

lossfun = torch.nn.MSELoss()


def train(model, device, train_loader, optimizer, epoch, train_loss, lenght=args.signal_lenght):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = auc_loss(output, y)
        loss.backward()
        optimizer.step()

        if i == 0:
            trainY = y.cpu().detach().numpy()
            pred_trainY = output.cpu().detach().numpy()
        else:
            trainY = np.vstack((trainY, y.cpu().detach().numpy()))
            pred_trainY = np.vstack((pred_trainY, output.cpu().detach().numpy()))

        if i % 5 == 0:
            print('{}s ECG Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(lenght,
                epoch, i*len(x), len(train_loader.dataset),
                100.*i/len(train_loader), loss.item()
            ))
            

    train_loss.append(loss.item())
      
    return trainY, pred_trainY


from sklearn.metrics import roc_curve, roc_auc_score
def roc_thr(trainY,pred_trainY):
     # tpr = Se,fpr = 1-Sp Youden_index = Se+Sp-1 = tpr-fpr
    fpr,tpr,thresholds = roc_curve(trainY,pred_trainY,1,drop_intermediate=False)
    Youden_index = tpr-fpr
    
    thr = thresholds[np.where(Youden_index == np.max(Youden_index))]
    return thr




def test(model, device, test_loader, thr, test_loss=[], lenght=args.signal_lenght, exam=False):
    model.eval()
    correct = 0 
    Pos = 0 
    Neg = 0 
    TP = 0
    TN = 0
    sample = 0
    sum_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.type(torch.float32)
            y = y.type(torch.float32)
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = auc_loss(output, y)
            prod = output.cpu().numpy()
            pred = (prod >= thr).astype('int8').reshape(-1)
            
            y_data = y.cpu().numpy().reshape(-1)
            Pos += len(np.argwhere(y_data == 1))
            Neg += len(np.argwhere(y_data == 0))
            sample += len(y_data)
            
            # acc
            correct += len(np.argwhere(pred == y_data))
            # se
            TP += len(np.argwhere(pred[np.argwhere(y_data == 1)] == 1))
            TN += len(np.argwhere(pred[np.argwhere(y_data == 0)] == 0))
            if i == 0:
                test_y = y_data
                test_pred = pred
            else:
                test_y = np.hstack((test_y, y_data))
                test_pred = np.hstack((test_pred, pred))
            sum_loss += loss.item()
    sum_loss /= (i+1)
    test_acc = 100.*correct/sample
    test_se = TP/Pos
    test_sp = TN/Neg
    test_ber = 1-0.5*(test_se+test_sp)
    test_auc = roc_auc_score(test_y, test_pred)
    if not exam:
        print('{}s ECG Test set: thr:{:.2f}  loss:{:.6f}'.format(lenght,thr[0],sum_loss))
        print('Evaluation: Acc:{:.2f}%   Se:{:.2f}%   Sp:{:.2f}%   Ber:{:.3f}%   Auc:{:.3f}'.format(test_acc, 
                                                                                                    100.*test_se,
                                                                                                    100.*test_sp,
                                                                                                    100.*test_ber,
                                                                                                    test_auc))
        test_loss.append(sum_loss)

        # save model
        min_loss = min(test_loss)
        if sum_loss == min_loss:
            print("save model ... ... \n")
            torch.save(model.state_dict(), r'./model//'+str(args.signal_lenght)+'s-'+str(k+1)+'-fold.pkl')
            np.save(r'./thr//'+str(args.signal_lenght)+'s-'+str(k+1)+'-fold.npy', thr)
    else:
        print('{}s-ECG Evaluation:'.format(lenght))
        print('Acc:{:.2f}%, Se:{:.2f}%, Sp:{:.2f}%, Ber:{:.3f}%, Auc:{:.3f}'.format(test_acc, 
                                                                                      100.*test_se,
                                                                                      100.*test_sp,
                                                                                      100.*test_ber,
                                                                                      test_auc))

    return test_pred, test_acc, test_se, test_sp, test_ber, test_auc



import os
import matplotlib.pyplot as plt
import scipy.io as sio



def main(args, k, index_train, index_test):
    train_loss = []
    test_loss = []

    # load data
    data = sio.loadmat(r'./data/public_ECG_5fold_data_with_annotion_1.mat')['data']
    # get train and test data
    trainDataset = TrainData(data, index_train)
    train_loader = DataLoader(dataset=trainDataset, batch_size=args.Train_batch_size, shuffle=True)
    testDataset = TestData(data, index_test)
    test_loader = DataLoader(dataset=testDataset, batch_size=args.Test_batch_size, shuffle=True)

    # create model
    mynet = CNN_LSTM().to(device)
    if args.auto_Lr:
        opt = torch.optim.Adam(mynet.parameters(), lr=args.Lr)
        torch.optim.lr_scheduler.ExponentialLR(opt, args.decay)
    else:
        opt = torch.optim.Adam(mynet.parameters(), lr=args.Lr)
    
    
    for epoch in range(args.Epoch):
        trainY, pred_trainY = train(mynet, device, train_loader, opt, epoch, train_loss)
        thr = roc_thr(trainY, pred_trainY)
        test_pred, test_acc, test_se, test_sp, test_ber, test_auc = test(mynet, device, test_loader, 
                                                                         thr, test_loss=test_loss)

    # test
    test_net = CNN_LSTM().to(device)
    test_net.load_state_dict(torch.load(r'./model//'+str(args.signal_lenght)+'s-'+str(k+1)+'-fold.pkl'))
    p_trainY, tr_acc, tr_se, tr_sp, tr_ber, tr_auc = test(test_net, device, train_loader, thr=0.5, exam=True)
    thr_test = np.load(r'./thr//'+str(args.signal_lenght)+'s-'+str(k+1)+'-fold.npy')
    p_testY, ts_acc, ts_se, ts_sp, ts_ber, ts_auc = test(test_net, device, test_loader, thr=thr_test, exam=True)
    
    # save important_data
    important_data = {'train_loss':train_loss,'test_loss':test_loss,
                      'Se':ts_se,'Sp':ts_sp,'Auc':ts_auc,'Ber':ts_ber,
                      'Se_tr':tr_se,'Sp_tr':tr_sp,'Auc_tr':tr_auc,'Ber_tr':tr_ber}
    data_path = r'./evaluation//'+str(args.signal_lenght)+'s-'+str(k+1)+'-fold.pickle'
    with open(data_path, 'wb') as fw:
        pickle.dump(important_data, fw)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    for k in range(1): 
        # train index
        index_train = np.argwhere(cv[k,]==0).reshape(-1)
        # test index
        index_test = np.argwhere(cv[k,]==1).reshape(-1)
        print('\n'+str(k+1)+'-fold train star ... ...')
        main(args, k, index_train, index_test)
    
    # show result
    Se = []; Sp = []; Ber = []; Auc = []
    for k in range(5):
        data_path = r'./evaluation//'+str(args.signal_lenght)+'s-'+str(k+1)+'-fold.pickle'
        with open(data_path, 'rb') as f:
            a = pickle.load(f)
            Se.append(a['Se']);Sp.append(a['Sp']);Ber.append(a['Ber']);Auc.append(a['Auc'])
    print('\n %ds ecg mean_Se:%.2f%%'%(args.signal_lenght, 100*np.mean(Se)))
    print(' %ds ecg mean_Sp:%.2f%%'%(args.signal_lenght, 100*np.mean(Sp)))
    print(' %ds ecg mean_Ber:%.2f%%'%(args.signal_lenght, 100*np.mean(Ber)))
    print(' %ds ecg mean_Auc:%.3f'%(args.signal_lenght, np.mean(Auc)))
    print('\n',end="")
    
