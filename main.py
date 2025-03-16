# This is an example to train a two-classes model.
import torch
import Encode_seq, Model
from multiprocessing import Process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def foo(i):
        print(" This is Process ", i)


def main():
        for i in range(5):
                p = Process(target=foo, args=(i,))
                p.start()


if __name__ == '__main__':
      #####################    Train   ########################
    main()

    data = Encode_seq.Biodata(fasta_file="acceptor.fasta",
                           label_file="acceptor_label.txt"
                           )
    dataset = data.process()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model.GCNNet().to(device)
    Model.train(dataset, model)


    ##############   Test    ####################
    data2 = Encode_seq.Biodata(fasta_file="acceptor.fasta",
                           label_file="acceptor_label.txt"
                           )

    data1 = data2.process()
    Model.test(data1)
