from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class New_GCN(nn.Module):

    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):  # 32 32
        super(New_GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in   # 96
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        """
        x: B c_in dim timestep  64 32 307 12
        support: [(B dim dim), ...]
        """
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# Static GCN w/ dense adj
class GCN(nn.Module):
    def __init__(self, K:int, input_dim:int, hidden_dim:int, bias=True, activation=nn.ReLU, weight_dim=69):
        super().__init__()
        self.K = K  # 3
        self.input_dim = input_dim   # 14
        self.hidden_dim = hidden_dim  # 14
        self.bias = bias
        self.activation = activation() if activation is not None else None
        self.init_params(n_supports=K)
        self.weight = nn.Parameter(torch.FloatTensor(weight_dim, K, input_dim, hidden_dim))
        nn.init.xavier_normal_(self.weight)

    def init_params(self, n_supports:int, b_init=0):
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, A:torch.Tensor, x:torch.Tensor, conv_E=None):
        node_num = x.shape[1]
        # A 是静态图
        A = A[1]
        conv_E_l = torch.mm(A, conv_E)
        supports = F.softmax(F.relu(torch.mm(conv_E_l, conv_E.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]  # [I, D-1/2AD-1/2]
        for k in range(2, self.K):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)  # (cheb_k, N, N)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # (B, cheb_k, N, seq)\
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, seq

        x_gconv = torch.einsum('bnki,nkio->bno', x_g, self.weight) + self.b  # b, N, seq
        return x_gconv  # (32, 69, 14)

    def __repr__(self):
        return self.__class__.__name__ + f'({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})'


class Adj_Preprocessor(object):
    def __init__(self, kernel_type:str, K:int):
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != 'localpool' else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, adj:torch.Tensor):
        '''
        Generate adjacency matrices
        :param adj: input adj matrix - (N, N) torch.Tensor
        :return: processed adj matrix - (K_supports, N, N) torch.Tensor
        '''
        kernel_list = list()

        if self.kernel_type in ['localpool', 'chebyshev']:  # spectral
            adj_norm = self.symmetric_normalize(adj)
            adj_norm = torch.where(torch.isnan(adj_norm), torch.full_like(adj_norm, 0), adj_norm)
            # adj_norm = self.random_walk_normalize(adj)     # for asymmetric normalization
            if self.kernel_type == 'localpool':
                localpool = torch.eye(adj_norm.shape[0]) + adj_norm  # same as add self-loop first
                kernel_list.append(localpool)

            else:  # chebyshev
                laplacian_norm = torch.eye(adj_norm.shape[0]) - adj_norm
                rescaled_laplacian = self.rescale_laplacian(laplacian_norm)
                kernel_list = self.compute_chebyshev_polynomials(rescaled_laplacian, kernel_list)

        elif self.kernel_type == 'random_walk_diffusion':  # spatial

            # diffuse k steps on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)
            '''
            # diffuse k steps bidirectionally on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            P_backward = self.random_walk_normalize(adj.T)
            forward_series, backward_series = [], []
            forward_series = self.compute_chebyshev_polynomials(P_forward.T, forward_series)
            backward_series = self.compute_chebyshev_polynomials(P_backward.T, backward_series)
            kernel_list += forward_series + backward_series[1:]  # 0-order Chebyshev polynomial is same: I
            '''
        else:
            raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion].')

        # print(f"Minibatch {b}: {self.kernel_type} kernel has {len(kernel_list)} support kernels.")
        kernels = torch.stack(kernel_list, dim=0)

        return kernels

    @staticmethod
    def random_walk_normalize(A):   # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)   # OD matrix Ai,j sum on j (axis=1)
        d_inv[torch.isinf(d_inv)] = 0.
        D = torch.diag(d_inv)
        A_norm = torch.mm(D, A)
        return A_norm

    @staticmethod
    def symmetric_normalize(A):
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = torch.mm(torch.mm(D, A), D)
        return A_norm

    @staticmethod
    def rescale_laplacian(L):
        # rescale laplacian to arccos range [-1,1] for input to Chebyshev polynomials of the first kind
        try:
            lambda_ = torch.linalg.eig(L)[0][:,0]      # get the real parts of eigenvalues
            lambda_max = lambda_.max()      # get the largest eigenvalue
        except:
            print("Eigen_value calculation didn't converge, using max_eigen_val=2 instead.")
            lambda_max = 2
        L_rescale = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescale

    def compute_chebyshev_polynomials(self, x, T_k):
        # compute Chebyshev polynomials up to order k. Return a list of matrices.
        # print(f"Computing Chebyshev polynomials up to order {self.K}.")
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k-1]) - T_k[k-2])
        return T_k


