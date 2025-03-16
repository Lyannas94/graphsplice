from torch import nn
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_geometric.nn as pyg_nn
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
import random
import matplotlib as plt
from sklearn.metrics import roc_curve, roc_auc_score
import math


class GCNNet(nn.Module):
    def __init__(self, K=2, node_hidden_dim=3, gcn_dim=128, gcn_layer_num=2, cnn_dim=16, cnn_layer_num=3, cnn_kernel_size=8, fc_dim=100, dropout_rate=0.3):
        super(GCNNet, self).__init__()
        self.K = K
        self.node_hidden_dim = node_hidden_dim
        self.gcn_dim = gcn_dim
        self.gcn_layer_num = gcn_layer_num
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(dropout_rate)   # Dropout
        self.relu = nn.ReLU()        # 激活函数
        self.embedding = nn.Embedding(self.K, self.node_hidden_dim)    # 嵌入层

        # 图卷积层
        self.gcn_layer = nn.ModuleList()
        for l in range(self.gcn_layer_num):
            if l == 0:
                self.gcn_layer.append(pyg_nn.SAGEConv((self.node_hidden_dim,self.node_hidden_dim), self.gcn_dim))
            else:                                   
                self.gcn_layer.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
            # self.gcn_layers1.append(SAGEConv(node_hidden_dim, self.gcn_dim))
            # node_hidden_dim = gcn_dim
        self.lns = nn.ModuleList()
        for l in range(self.gcn_layer_num-1):
            self.lns.append(nn.LayerNorm(self.gcn_dim))
        # 卷积层
        self.cnn_layer = nn.ModuleList()
        for _ in range(cnn_layer_num):
            self.cnn_layer.append(nn.Conv1d(self.node_hidden_dim, self.cnn_dim, self.cnn_kernel_size))
            self.cnn_dim = self.cnn_dim * 2

        # # 全连接层
        # self.fc = nn.Linear(node_hidden_dim, fc_dim)

        # # 输出层
        # self.out = nn.Linear(fc_dim, 2)  # 假设我们有2个类

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = self.embedding(x)
        for i in range(self.gcn_layer_num):
            x = self.gcn_layer[i](x, edge_index)
            x = self.relu(x,)
            x = self.dropout(x,p=self.dropout_rate,training=self.training)
            if not i == self.gcn_layer_num - 1:
                x = self.lns[i](x)

        for i in range(self.cnn_layer_num):
            x = self.cnn_layer[i](x)
            x = self.relu(x,)
            if not i == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            # x = self.dropout(x, p=self.dropout, training=self.training)

        # flatten layer
        x = x.flatten(start_dim = 1)
        x = self.d1(x)
        x = self.relu(x)
        x = self.d2(x)
        out = self.softmax(x,dim = 1)

        return out

def train(dataset, model, learning_rate=1e-4, batch_size=64, epoch_n=15,
          random_seed=200, val_split=0.2, model_name="model.pt",
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    torch.manual_seed(random_seed)

    #split train dataset and val dataset
    data_list = list(range(0, len(dataset)))


    test_list = random.sample(data_list, int(len(dataset) * val_split))
    train_dataset = [dataset[i] for i in data_list if i not in test_list]
    val_dataset = [dataset[i] for i in data_list if i in test_list]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    # train
    model = model.to(device)
    old_train_acc = 0
    old_val_acc = 0

    for epoch in range(epoch_n):
        training_running_loss = 0.0
        train_acc = 0.0
        model.train()
        for data in train_loader:
            data = data.to(device)
            label = data.y

            # forward + backprop + loss
            pred = model(data)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()

             # update model params
            optimizer.step()
            training_running_loss += loss.detach().item()
            train_acc += (pred.argmax(dim = 1) == label).type(torch.float).sum().item()

        # test accuracy
        val_acc = evaluation(val_loader, model, device)
        if train_acc > old_train_acc:
            torch.save(model, model_name)
            old_train_acc = train_acc
        print("Epoch {}| Loss: {:.4f}| Train accuracy: {:.4f}| Validation accuracy: {:.4f}".format(epoch, training_running_loss/(i+1), train_acc/(i+1), val_acc))
    
    return model


 # evaluate
def evaluation(loader, model, device):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
        correct += pred.eq(label).sum().item()
    total = len(loader.dataset)
    acc = correct / total
    return acc

def test(data1, model_name="model.pt", val_split=0.2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    data_list = list(range(0, len(data1)))
    test_list = random.sample(data_list, int(len(data1) * val_split))
    print(len(test_list))

    testset = [data1[i] for i in data_list if i in test_list]
    model = torch.load(model_name, map_location=device)
    loader = DataLoader(testset, batch_size=len(data1), shuffle=False, follow_batch=['x_src', 'x_dst'])
    model.eval()

    TP, FN, FP, TN = 0, 0, 0, 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
            # correct += pred.eq(label).sum().item()
            A, B, C, D = eff(label, pred)
            TP += A
            FN += B
            FP += C
            TN += D
            AUC = Calauc(label, pred)
    SN, SP, ACC, MCC, F1Score, PRE, Err = Judeff(TP, FN, FP, TN)
    print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
    print("SN: {:.4f}, SP: {:.4f}, ACC: {:.4f}, MCC: {:.4f}, AUC: {:.4f}, F1Score: {:.4f}, PRE: {:.4f}, Err: {:.4f}".format(SN, SP, ACC, MCC, AUC, F1Score, PRE, Err))

def eff(labels, preds):

    TP, FN, FP, TN = 0, 0, 0, 0

    for idx, label in enumerate(labels):

        if label == 1:
            if label == preds[idx]:
                TP += 1
            else:
                FN += 1
        elif label == preds[idx]:
            TN += 1
        else:
            FP += 1

    return TP, FN, FP, TN

def Judeff(TP, FN, FP, TN):

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    MCC = (TP * TN - FP * FN) / (math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    F1Score = (2 * TP) / (2 * TP + FN + FP)
    PRE = TP / (TP + FP)
    Err = 1 - ((TP + TN) / (TP + FN + FP + TN))

    return SN, SP, ACC, MCC, F1Score, PRE, Err

def Calauc(labels, preds):

    labels = labels.clone().detach().cpu().numpy()
    preds = preds.clone().detach().cpu().numpy()

    # f = list(zip(preds, labels))
    # rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    # rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    # pos_cnt = np.sum(labels == 1)
    # neg_cnt = np.sum(labels == 0)
    # AUC = (np.sum(rankList) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)
    fpr, tpr, thresholds = roc_curve(labels, preds)
    AUC = roc_auc_score(labels, preds)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    return AUC













