import numpy as np
from collections import defaultdict, deque

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.shared_base import *
from utils import get_logger, get_variable, keydefaultdict

logger = get_logger()

def isnan(tensor):
    return np.isnan(tensor.cpu().data.numpy()).sum() > 0


# https://github.com/carpedm20/ENAS-pytorch/blob/master/models/shared_rnn.py

class NAS_RNN(SharedModel):
    def __init__(self, args, input_size, hidden_size, num_blocks):
        super(NAS_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.h_varient = args.hidden_varient 
        self.args = args

        self.w_xh = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.w_xc = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)

        self.w_h, self.w_c = defaultdict(dict), defaultdict(dict)

        for idx in range(self.num_blocks):
            for jdx in range(idx+1, self.num_blocks):
                self.w_h[idx][jdx] = nn.Linear(
                        self.hidden_size, self.hidden_size, bias=False)
                self.w_c[idx][jdx] = nn.Linear(
                        self.hidden_size, self.hidden_size, bias=False)

        self._w_h = nn.ModuleList(
                [self.w_h[idx][jdx] for idx in self.w_h for jdx in self.w_h[idx]])
        self._w_c = nn.ModuleList(
                [self.w_c[idx][jdx] for idx in self.w_c for jdx in self.w_c[idx]])

        if self.args.use_batchnorm:
            self.batch_norm = nn.BatchNorm1d(hidden_size, momentum=self.args.batchnorm_momentum)




    def forward(self, x, h_prev, dag):
        # if h_variant value is lstm when h_prev is a state information of [m,h], where m is memory
        c, h, f, outputs = {}, {}, {}, {}
        
        f[0] = self.get_f(dag[-1][0].name)
        c[0] = F.sigmoid(self.w_xc(t.cat([x, h_prev], -1)))

        if self.args.use_highway_connections:
            h[0] = c[0] * f[0](self.w_xh(t.cat([x, h_prev], -1))) + (1 - c[0]) * h_prev
        else:
            h[0] = f[0](self.w_xh(t.cat([x, h_prev], -1)))


        leaf_node_ids = []
        q = deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.num_blocks:
                    leaf_node_ids.append(node_id)
                    assert len(nodes) == 1, "parent of leaf node should have only one child"
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                f[next_id] = self.get_f(next_node.name)
                if self.args.use_highway_connections:
                    c[next_id] = F.sigmoid(w_c(h[node_id]))
                    h[next_id] = c[next_id] * f[next_id](w_h(h[node_id])) + (1 - c[next_id]) * h[node_id]
                else:
                    h[next_id] = f[next_id](w_h(h[node_id]))

                q.append(next_id)

        # average all the loose ends
        leaf_nodes = [h[node_id] for node_id in leaf_node_ids]
        output = t.mean(t.stack(leaf_nodes, 2), -1)

        if self.args.use_batchnorm:
            output = self.batch_norm(output)


        
        if self.h_varient == 'simple':
            return output, h[self.num_blocks-1]
        elif self.h_varient == 'gru':
            return output, output
        else:
            raise Exception(f"Unknown h_varient:{h_varient} !!")

    def init_hidden(self, batch_size):
        zeros = t.zeros(batch_size, self.hidden_size)
        return get_variable(zeros, requires_grad=False)

    def get_f(self, name):
        name = name.lower()
        if name == 'relu':
            f = F.relu
        elif name == 'tanh':
            f = F.tanh
        elif name == 'identity':
            f = lambda x: x
        elif name == 'sigmoid':
            f = F.sigmoid
        return f

    def get_num_cell_parameters(self, dag):
        num = 0

        num += size(self.w_xc)
        num += size(self.w_xh)

        q = deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.num_blocks:
                    assert len(nodes) == 1, "parent of leaf node should have only one child"
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                num += size(w_h)
                num += size(w_c)

                q.append(next_id)

        logger.debug(f"# of cell parameters: {format(self.num_parameters, ',d')}")
        return num

    def reset_parameters(self):
        init_range = 0.025 if self.args.mode == 'train' else 0.04
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)

