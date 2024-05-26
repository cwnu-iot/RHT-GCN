import os
import time
import numpy as np
from CRHT_DGCN import CRHT_DGCN
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.autograd import Variable

device = torch.device('cuda')
path='danrenshuju'
allpath=os.listdir(path)
print(allpath)
BATCH_SIZE = 64

def getdata(path):
    alldata = []
    maxdf, maxrssi, mindf = 0, 0, 0
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            x = line.split()
            a = float(x[3])
            b = float(x[4])
            if a > maxdf:
                maxdf = a
            if a < mindf:
                mindf = a
            if b < maxrssi:
                maxrssi = b
    with open(path, "r") as f:
        # print(path)
        for line in f.readlines():
            data = []
            line = line.strip('\n')
            x = line.split()
            data.append(int(x[1]))
            mmm = float(x[3])
            if mmm > 0:
                mmm = mmm / maxdf
            if mmm < 0:
                mmm = mmm / mindf
            xinh = float(x[4]) / maxrssi
            data.append(mmm)
            data.append(xinh)
            data.append(float(x[5]) / 6.280117345603815)
            alldata.append(data)
    newdata = []
    for k in range(16):
        newdata.append([])
    for j in alldata:
        a = j[0]
        newdata[a].append(j[1:4])
    tudata = []
    for i in newdata:
        #print("i",len(i))
        danchongdao = []
        if len(i) <= 84:
            zerolist = [0] * 3
            for j in range(84 - len(i)):
                i.append(zerolist)
            tudata.append(i)
        if len(i) > 84:
            danchongdao = i[:84]
            tudata.append(danchongdao)
       # # 获得邻接矩阵
    # start_node = list()
    # end_node = list()
    # with open(adj_filename, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         content = line.rstrip().split(',')
    #         start_node.append(int(content[0]))
    #         end_node.append(int(content[1]))
    # #print(start_node)
    # #print(end_node)
    #
    # newnode = []
    # adjlist = []
    # heynode = []
    # for i,z in enumerate(tudata):
    #     # for g in z:
    #         #for i in range(len(g)):
    #     if z[0] == [0]*3:
    #         # 当节点i的数据长度为0，则用节点i的邻接节点数据的均值来表示节点i
    #         for j in range(len(start_node)):
    #             if i == start_node[j]:
    #                 adjlist.append(tudata[end_node[j]])
    #         # 求平均
    #         for items in zip(*adjlist):
    #             for one in zip(*items):
    #                 heynode.append(sum(one) / len(one))
    #             newnode.append(heynode)
    #             heynode = []
    #         tudata[i] = newnode
    #         newnode = []
    #         adjlist = []
    reshapedata = []
    for i in range(84):
        a = []
        for j in range(16):
            a.append(tudata[j][i])
        reshapedata.append(a)
    return reshapedata

def data_lodar(path):
    datas = []
    labels = []
    path1 = os.listdir(path)
    for i in path1:
        path2 = os.path.join(path, str(i))
        path3 = os.listdir(path2)
        for j in path3:
            label = j.split('_')[1][:-4]
            labels.append(label)
            path4 = os.path.join(path2, j)
            data = getdata(path4)
            datas.append(data)
    print('label', len(labels), type(labels))
    print('data', len(datas), type(datas))
    return datas, labels

datas,labels=data_lodar(path)


import random
def split_train_test(data,label,test_ratio):
    #设置随机数种子，保证每次生成的结果都是一样的

    '''random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    print(test_set_size)
    test_data = torch.Tensor(data[:test_set_size])
    test_label = torch.Tensor(label[:test_set_size])
    train_data = torch.Tensor(data[test_set_size:])
    train_label = torch.Tensor(label[test_set_size:])'''

    random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    newdata,newlabel=[],[]
    for k in range(21):
        newdata.append([]),newlabel.append([])
    for i in range(len(label)):
        newdata[int(label[i])].append(data[i])
    train_data,test_data,train_label,test_label=[],[],[],[]
    #print('ssr',len(newdata[0]),len(label[0]),len(label),len(newdata))
    for i in range(len(newdata)):
        for j in range(len(newdata[i])):
            if j <int(len(newdata[i])*test_ratio):
                test_data.append(newdata[i][j])
                test_label.append(i)
            else:
                train_data.append(newdata[i][j])
                train_label.append(i)
    random.seed(42)
    random.shuffle(train_data)
    random.seed(42)
    random.shuffle(train_label)
    '''test_data = np.array(test_data)
    test_label=np.array(test_label)
    train_data = np.array(train_data)
    train_label=np.array(train_label)'''
    test_data = torch.Tensor(test_data)
    test_label = torch.Tensor(test_label)
    train_data = torch.Tensor(train_data)
    train_label = torch.Tensor(train_label)
    #iloc选择参数序列中所对应的行
    return train_data,train_label,test_data,test_label

traindata,trainlabel,testdata,testlabel=split_train_test(datas,labels,0.2)
from torch.utils.data import Dataset, DataLoader, TensorDataset

traindata = TensorDataset(traindata, trainlabel)
testdata = TensorDataset(testdata, testlabel)
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.001

def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    """
    Calculate confusion matrix on the provided preds and labels.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
        num_classes (int): number of classes. Tensor(s) must be on CPU.
        normalize (Optional[str]) : {‘true’, ‘pred’, ‘all’}, default="true"
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix
            will not be normalized.
    Returns:
        cmtx (ndarray): confusion matrix of size (num_classes x num_classes)
    """
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))#, normalize=normalize) 部分版本无该参数
    return cmtx

rnn=CRHT_DGCN(num_class=21, num_point=16, num_person=1, in_channels=3,
                 drop_out=0, adaptive=True).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(dataset=traindata, batch_size=BATCH_SIZE, shuffle=True)
def train(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
            b_y=torch.as_tensor(b_y,dtype=torch.int64)    #x1.size torch.Size([32, 3, 32, 16])
            #print('b_x.size()',b_x.size())
            b_x=b_x.permute(0,3,1,2)
            b_x=b_x.unsqueeze(-1)
            #print(b_x.size())
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = rnn(b_x)  # rnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
best_acc =[]
cmtxlist = []
def test(ep):
    test_loss = 0
    correct = 0
    preds = []
    labels = []
    # data, label = data0,label0
    test_data = DataLoader(testdata, BATCH_SIZE, shuffle=True)
    if ep == 1:
        torch.save(rnn.state_dict(), 'CRHT-DGCN.mdl')
    with torch.no_grad():
        for data, target in test_data:
            target = torch.tensor(target, dtype=torch.int64)
            data, target = data.to(device), target.to(device)
            data = data.permute(0, 3, 1, 2)
            data = data.unsqueeze(-1)
            data, target = Variable(data, volatile=True), Variable(target)
            output = rnn(data)
            preds.append(output.cpu())
            labels.append(target.cpu())
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_data.dataset)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        cmtx = get_confusion_matrix(preds, labels, 21)
        print(cmtx.tolist())
        # # print('============================')
        # # print(len(test_data.dataset))
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        #     test_loss, correct, len(test_data.dataset),
        #     100. * correct / len(test_data.dataset)))
        # acc = correct / len(test_data.dataset)
        # best_acc.append(float(acc))
        # print(best_acc)
        # print(max(best_acc))
        # return test_loss
        for i in range(len(cmtx)):
            print(cmtx[i][i]/sum(cmtx[i]))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_data.dataset),
            100. * correct / len(test_data.dataset)))
        acc = correct / len(test_data.dataset)
        best_acc.append(float(acc))
        cmtxlist.append(cmtx.tolist())
        if ep == 49:
            with open("CRHT-DGCN", "a+") as f:
                f.write('\n')
                f.write(str(best_acc))
                f.write('\n')
                f.write(str(max(best_acc)))
                f.write(str(cmtxlist[best_acc.index(max(best_acc))]))
                f.write('\n')
        print('acc',best_acc)
        print(max(best_acc))
        return test_loss
if __name__ == '__main__':
    begintime=time.time()
    for epoch in range(0, 50):
        print(epoch)
        train(epoch)
        Stime = time.time()
        test(epoch)
        Etime = time.time()
        print("ctime=", (Etime - Stime)/40)
        if epoch %15==0:
            LR /= 10
    endtime=time.time()
    print('time',endtime-begintime)
