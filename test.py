from Bio import SeqIO
import numpy as np
# dna_seq = {}
# for seq_record in SeqIO.parse(".\Datasets\\Worm\\sequences_donor_400.fasta", "fasta"):
#       # 读取 fasta 文件
#     dna_seq[seq_record.id] = str(seq_record.seq)

# seq_list = list(dna_seq.values())
# # print(seq_list)

# # label = np.loadtxt(".\Datasets\\Worm\\labels_acceptor.txt")
# # print(label)
# print(seq_list[1])


# def num_transfer_loc(num_seq, K):
#     loc = []
#     for i in range(0, len(num_seq)-K+1):
#         loc.append(int(num_seq[i:i+K], 4))
    
#     return loc


# num_seq = "0123"
# K = 3
# location = ""
# location = num_transfer_loc(num_seq, K)
# print(location)
