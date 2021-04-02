from abc import ABC

from graph_ae.Layer import SGAT
from torch_sparse import spspmm, coalesce
from graph_ae.SAGEConv import SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import sort_edge_index, add_remaining_self_loops
import torch.nn.functional as F
import torch

device = torch.device('cuda:1')


class Net(torch.nn.Module, ABC):
    model = None
    optimizer = None

    cfg = {'lr': 1e-3,
           'betas': (0.9, 0.999)}

    def __init__(self):
        super(Net, self).__init__()
        out = 500
        size = 2
        self.depth = 3
        rate = [0.8, 0.8, 0.8]
        # rate = [0.6, 0.6, 0.6]
        shape = [64, 64, 64]
        self.direction = 1
        self.down_list = torch.nn.ModuleList()
        self.up_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()
        # encoder
        conv = SGAT(size, out, shape[0])
        self.down_list.append(conv)
        for i in range(self.depth - 1):
            pool = TopKPooling(shape[i], rate[i])
            self.pool_list.append(pool)
            conv = SGAT(size, shape[i], shape[i + 1])
            self.down_list.append(conv)
        pool = TopKPooling(shape[-1], rate[-1])
        self.pool_list.append(pool)

        # decoder
        for i in range(self.depth - 1):
            conv = SAGEConv(shape[self.depth - i - 1], shape[self.depth - i - 2])
            self.up_list.append(conv)
        conv = SAGEConv(shape[0], out)
        self.up_list.append(conv)

        self.x_num_nodes = 0
        self.x = 0
        self.r_graph = 0

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.model = Net()
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.model

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        return edge_index, edge_weight

    def forward(self, data):
        x, edge_index, y, batch = data.x, data.edge_index, data.y, data.batch
        self.x = x

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])

        edge_list = []
        perm_list = []
        shape_list = []
        edge_weight = None

        f, e, b = x, edge_index, batch
        for i in range(self.depth):
            if i < self.depth:
                edge_list.append(e)
            f, attn = self.down_list[i](f, e, self.direction)
            shape_list.append(f.shape)
            f = F.leaky_relu(f)
            f, e, _, b, perm, _ = self.pool_list[i](f, e, edge_weight, b, attn)
            if i < self.depth - 1:
                e, _ = self.augment_adj(e, None, f.shape[0])
            perm_list.append(perm)
        latent_x, latent_edge = f, e

        z = f
        for i in range(self.depth):
            index = self.depth - i - 1
            shape = shape_list[index]
            up = torch.zeros(shape).to(device)
            p = perm_list[index]
            up[p] = z
            z = self.up_list[i](up, edge_list[index])
            if i < self.depth - 1:
                z = torch.relu(z)

        edge_list.clear()
        perm_list.clear()
        shape_list.clear()

        return z, latent_x, latent_edge, b

    def train_model(self, train_set, train_set2, valid_set, num_epoch, m_name):
        model = self.model
        optimizer = self.optimizer
        mse_list = []
        num_nodes_list = []
        total_loss_list = []
        tmp_list = []

        for e in range(num_epoch):
            reconstruction_loss = 0
            reconstruction_loss_1 = 0
            nodes_num = 0
            r_graph = 0
            for data in train_set:
                optimizer.zero_grad()
                data = data.to(device)
                z, _, _, _ = model(data)

                mse_loss = torch.nn.MSELoss()(z, data.x)
                mix_loss = mse_loss
                mix_loss.backward()
                reconstruction_loss += mse_loss.item()

                optimizer.step()
                nodes_num = 0
                r_graph = 0

            for data in valid_set:
                data = data.to(device)
                z, _, _, _ = model(data)
                mse_loss = torch.nn.MSELoss()(z, data.x)
                reconstruction_loss_1 += mse_loss.item()

            reconstruction_loss /= len(train_set)
            reconstruction_loss_1 /= len(valid_set)

            if e >= 0:
                mse_list.append(reconstruction_loss)
                num_nodes_list.append(nodes_num)
                total_loss_list.append(reconstruction_loss)
                tmp_list.append(reconstruction_loss_1)

            print()
            print('Epoch: {:03d}'.format(e))
            print('AVG Reconstruction Loss:', reconstruction_loss)
            print('AVG test Loss:', reconstruction_loss_1)
            print('Nodes num:', nodes_num)
            print('Remaining graph:', r_graph)

        torch.save(model.state_dict(), "../data/model/G/" + m_name)
        print("model saved")
        x1, e1, batch1 = None, None, None
        for data in train_set2:
            data = data.to(device)
            _, x1, e1, batch1 = model(data)

        x2, e2, batch2 = None, None, None
        for data in valid_set:
            data = data.to(device)
            _, x2, e2, batch2 = model(data)

        return [mse_list, num_nodes_list, total_loss_list, tmp_list], \
               [x1.detach(), e1.detach(), batch1.detach()], \
               [x2.detach(), e2.detach(), batch2.detach()]
