import sys, pathlib
import os, glob, json
import torch
import numpy as np
import itertools
import tqdm
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from collections import OrderedDict
from utils.nb201_dataset import NasBench201Dataset
from utils.utils_data import train_valid_test_split_dataset


class Dataset:

    ##########################################################################
    def __init__(self, hp: str, nb201_seed: int):
        self.num_features = 7
        self.hp = hp
        if __name__ == "__main__":
            path = os.path.join("..", "data", "nasbench201_pg_ae")
        else:
            path = os.path.join("data", "nasbench201_pg_ae")  # for debugging

        pathlib.Path(path).mkdir(exist_ok=True)

        file_cache_train = os.path.join(path, "cache_train")
        file_cache_test = os.path.join(path, "cache_test")
        file_cache = os.path.join(path, "cache")
        ############################################

        if not os.path.isfile(file_cache):
            nasbench = NasBench201Dataset(start=0, end=15624, hp=hp, seed=nb201_seed)
            self.data = []
            for graph in tqdm.tqdm(nasbench):
                self.data.append(self.map_item(graph))
                map_network = Dataset.map_network(graph)
                self.data[-1].edge_index = map_network[0]
                self.data[-1].x = map_network[1]
                self.data[-1].num_nodes = graph.x.shape[0]

            print(f"Saving data to cache: {file_cache}")
            torch.save(self.data, file_cache)

        ############################################
        if not os.path.isfile(file_cache_train):
            print(f"Loading data from cache: {file_cache}")
            self.data = torch.load(file_cache)

            self.train_data, self.test_data = Dataset.sample(self.data)

            print(f"Saving train data to cache: {file_cache_train}")
            torch.save(self.train_data, file_cache_train)

            print(f"Saving test data to cache: {file_cache_test}")
            torch.save(self.test_data, file_cache_test)

        else:
            print(f"Loading train data from cache: {file_cache_train}")
            self.train_data = torch.load(file_cache_train)

            print(f"Loading test data from cache: {file_cache_test}")
            self.test_data = torch.load(file_cache_test)

        ############################################

        self.length = len(self.train_data) + len(self.test_data)

    ##########################################################################
    def map_item(self, graph):
        train_acc = graph.y[0, :]
        valid_acc = graph.y[1, :]

        train_acc = torch.FloatTensor(train_acc / 100.0)
        valid_acc = torch.FloatTensor(valid_acc / 100.0)

        if self.hp == '12':
            test_acc = graph.y[2, :]
            test_acc = torch.FloatTensor([test_acc / 100.0])
            return Data(train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)

        return Data(train_acc=train_acc, valid_acc=valid_acc)

    ##########################################################################
    @staticmethod
    def map_network(graph):
        node_attr = torch.FloatTensor(graph.x)
        edge_index = torch.tensor(np.nonzero(graph.a))

        return edge_index.long(), node_attr


    ##########################################################################
    @staticmethod
    def sample(dataset, shuffle=True, shuffle_seed=0):
        data_dict = train_valid_test_split_dataset(dataset, ratio=[0.9, 0.1], shuffle=shuffle, shuffle_seed=shuffle_seed)
        return data_dict['train'], data_dict['valid']

    ##########################################################################
    '''
    @staticmethod
    def pg_graph_to_nb201(pg_graph):
        # first tensor node attributes, second is the edge list
        ops = [OPS_by_IDX_201[i] for i in pg_graph.x.cpu().numpy()]
        matrix = np.array(to_dense_adj(pg_graph.edge_index)[0].cpu().numpy())
        try:
            if (matrix == ADJACENCY).all():
                steps_coding = ['0', '0', '1', '0', '1', '2']

                node_1 = '|' + ops[1] + '~' + steps_coding[0] + '|'
                node_2 = '|' + ops[2] + '~' + steps_coding[1] + '|' + ops[3] + '~' + steps_coding[2] + '|'
                node_3 = '|' + ops[4] + '~' + steps_coding[3] + '|' + ops[5] + '~' + steps_coding[4] + '|' + ops[
                    6] + '~' + steps_coding[5] + '|'
                nodes_nb201 = node_1 + '+' + node_2 + '+' + node_3
                index = nasbench.query_index_by_arch(nodes_nb201)
                acc = Dataset.map_item(index).acc
            else:
                acc = torch.zeros(1)
        except:
            acc = torch.zeros(1)

        return acc
    '''
    def to_TUDataset(self, root='', name='nb201'):
        datas = {'train': self.train_data, 'test': self.test_data}
        for key, value in datas.items():
            output_dir = pathlib.Path(os.path.join(root, f'{name}_{key}'))
            output_dir.mkdir(exist_ok=True)
            for data in DataLoader(value, batch_size=len(value), shuffle=False):
                np.savetxt(os.path.join(output_dir, f'{name}_{key}_node_attributes.txt'), data.x.numpy(), fmt='%i')
                edge_index = data.edge_index.T.to(torch.long).numpy()
                np.savetxt(os.path.join(output_dir, f'{name}_{key}_A.txt'), edge_index, fmt='%i')
                y = data.valid_acc.reshape((len(value), int(self.hp)))
                y = np.expand_dims(y[:, -1], -1)
                np.savetxt(os.path.join(output_dir, f'{name}_{key}_graph_labels.txt'), y)


##############################################################################
#
#                              Debugging
#
##############################################################################

if __name__ == "__main__":
    ds = Dataset('200', 777)
    print(ds)