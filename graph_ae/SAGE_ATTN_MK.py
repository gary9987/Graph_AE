from typing import Union, Tuple, Optional
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn import Parameter
from torch_geometric.nn import global_mean_pool as g_pooling


class SAGEAttn(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, num_k, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        super(SAGEAttn, self).__init__(aggr='mean', **kwargs)
        kwargs.setdefault('aggr', 'mean')
        self.num_k = num_k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.beta = Parameter(torch.Tensor(1))
        self.pm = Parameter(torch.ones([self.num_k]))
        self._alpha = None
        self.layers = torch.nn.ModuleList()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        for i in range(num_k):
            self.layers.append(Linear(in_channels[1], out_channels, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        self.beta.data.fill_(1)
        self.pm.data.fill_(1)
        self.lin_l.reset_parameters()
        for lin in self.layers:
            lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        x_norm = F.normalize(x, p=2., dim=-1)
        x1: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        x_r = x1[1]

        out = self.propagate(edge_index, x=x, x_norm=x_norm, size=None)
        out = self.lin_l(out)

        alpha = self._alpha
        self._alpha = None

        pm = torch.softmax(self.pm, dim=-1)
        if x_r is not None:
            for i in range(self.num_k):
                out += pm[i] * F.leaky_relu(self.layers[i](x_r))
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        e_batch = edge_index[0]
        node_scores = g_pooling(alpha, e_batch).view(-1)
        return out, node_scores

    def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self.beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
