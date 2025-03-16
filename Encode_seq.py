from torch_geometric.data import Data
import torch
import numpy as np
from Bio import SeqIO
from collections import Counter

# class BipartiteData(Data):   # Bipartite graph data
#     # 这个类继承了 PyTorch Geometric 的 Data 类，用于存储双分图数据
#     def _add_other_feature(self, other_feature) :  # 这个函数用于添加其他特征
#         self.other_feature = other_feature
#
#     def __inc__(self, key, value):  # 这个函数用于返回图中节点的数量
#         if key == 'edge_index':  # key是边的索引，value是边的数量
#             return torch.tensor([[self.x_src.size(0)], [self.x_dst.size(0)]])
#         # self.x_src.size(0) 是源节点特征矩阵中的节点数量
#         # self.x_dst.size(0) 是目标节点特征矩阵中的节点数量
#         # torch.tensor 是 PyTorch 中的张量
#         else:
#             return super(BipartiteData, self).__inc__(key, value)
#             # 这里是调用父类的 __inc__ 函数
class Biodata:
    def __init__(self, fasta_file, label_file, K = 3):
        self.fasta_file = fasta_file
        self.K = K
        self.label_file = label_file

    def graph(self, dna_seq):
        # 计算k-mer的频率
        k_mer_freq = Counter([dna_seq[i:i+self.K] for i in range(len(dna_seq) - self.K + 1)])
        # 创建节点特征和边索引列表
        node_features = []
        edge_index = []
        for i in range(len(dna_seq) - self.K + 1):
            k_mer = dna_seq[i:i+self.K]
            position = i
            # 添加k-mer节点，特征为频率
            node_features.append([k_mer_freq[k_mer], position])
            # # 添加位置节点，特征为位置
            # node_features.append([position, 0])
            # 添加边，特征为是否为经典位点
            if 'GT' in k_mer or 'AG' in k_mer or 'AT' in k_mer or 'AC' in k_mer or 'GC' in k_mer or 'AG' in k_mer:
                edge_index.append([2*i, 2*i+1])
            else:
                edge_index.append([0,0])
        edge_index = np.array(edge_index).T

        # 创建Data对象
        dna_graph = Data(x=torch.tensor(node_features, dtype=torch.long), edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous())
        # wwww
        return dna_graph

    def process(self):
    # 读取fasta文件
        dna_seq = {}
        for seq_record in SeqIO.parse(self.fasta_file, "fasta"):
            dna_seq[seq_record.id] = str(seq_record.seq)
        # 读取标签文件
        labels = np.loadtxt(self.label_file)
        # 处理图数据
        data_list = []
        for i, seq in enumerate(dna_seq.values()):
            graphdata = self.graph(seq)
            graphdata.y = torch.tensor([labels[i]], dtype=torch.long)
            data_list.append(graphdata)
            # wwww
        return data_list
            



    