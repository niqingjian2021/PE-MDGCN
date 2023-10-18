import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from GCN import GCN, New_GCN


class GTU(nn.Module):
    def __init__(self, in_channels, time_strides=1, kernel_size=3):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu


class ModuleBlock(nn.Module):
    def __init__(self, seq_len: int, n_nodes: int, input_dim: int,
                 lstm_hidden_dim: int, lstm_num_layers: int,
                 K: int, gconv_use_bias: bool, gconv_activation=nn.ReLU, device='cpu',
                 days=None, dims=None, num_nodes=None, is_dynamic=False
                 ):
        super().__init__()
        self.seq_len = seq_len  # 14 = 12 + 1 + 1
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim  # 64
        self.lstm_num_layers = lstm_num_layers  # 3

        self.fc = nn.Linear(in_features=seq_len, out_features=seq_len, bias=False)  # 14 14 [12 + 1 + 1]  观测长度
        self.fc2 = nn.Linear(in_features=seq_len, out_features=seq_len, bias=False)

        self.obs_start_conv = nn.Conv2d(in_channels=1,
                                        out_channels=32,
                                        kernel_size=(1, 1))
        self.obs_end_conv = nn.Conv2d(in_channels=32,
                                      out_channels=1,
                                      kernel_size=(1, 1))

        # self.conv_l_e = nn.Parameter(torch.randn(69, 69), requires_grad=True)
        # self.conv_r_e = nn.Parameter(torch.randn(69, 69), requires_grad=True)
        self.conv_E = nn.Parameter(torch.randn(n_nodes, 8), requires_grad=True).to(device)

        if is_dynamic:
            self.gconv_temporal_feats = New_GCN(c_in=32, c_out=32, dropout=0.3)
            self.start_conv = nn.Conv2d(in_channels=1,
                                        out_channels=32,
                                        kernel_size=(1, 1))
            self.end_conv = nn.Conv2d(in_channels=32,
                                      out_channels=1,
                                      kernel_size=(1, 1),
                                      bias=True)
            self.batch_norm = nn.BatchNorm2d(32)
        else:
            self.gconv_temporal_feats = GCN(K=K, input_dim=seq_len, hidden_dim=seq_len,
                                            bias=gconv_use_bias, activation=gconv_activation, weight_dim=n_nodes)

        self.gtu1 = GTU(32, 1, 1)  # T
        self.gtu3 = GTU(32, 1, 3)  # T - 2
        self.gtu5 = GTU(32, 1, 5)  # T - 3
        self.gtu7 = GTU(32, 1, 7)  # T - 6
        self.final_fc = nn.Sequential(
            nn.Linear(4 * seq_len - 12, 64),  # 4T - 12
            nn.Dropout(0.05),
        )

    def forward(self, adj: torch.Tensor, obs_seq: torch.Tensor, hidden: tuple, is_dynamic=False):
        batch_size = obs_seq.shape[0]
        obs_seq = obs_seq.permute(0, 3, 2, 1)
        x_seq = obs_seq.sum(dim=-1)  # sum up feature dimension: default 1  (32, 14, 69, 1)
        x_seq = x_seq.permute(0, 2, 1)  # x_seq (batch, nodes, seq_len)
        if not is_dynamic:
            x_seq_gconv = self.gconv_temporal_feats(A=adj, x=x_seq, conv_E=self.conv_E)
        else:
            tmp_x_seq = torch.unsqueeze(x_seq, dim=1)
            tmp_x_seq = self.start_conv(tmp_x_seq)  # (32, 32, 69, 14)
            x_seq_gconv = self.gconv_temporal_feats(tmp_x_seq, [adj])  # (32, 32, 69, 14)
            x_seq_gconv = self.end_conv(x_seq_gconv)
            x_seq_gconv = torch.squeeze(x_seq_gconv)

        x_hat = torch.add(x_seq, x_seq_gconv)
        z_t = x_hat.sum(dim=1) / x_hat.shape[1]  # agg
        s = torch.sigmoid(self.fc2(torch.relu(self.fc(z_t))))  # (32,14)
        obs_seq_expand = obs_seq.permute(0, 3, 2, 1)
        obs_seq_expand = self.obs_start_conv(obs_seq_expand)
        obs_seq_reweighted = torch.einsum('bfnt,bt->bfnt',
                                          [obs_seq_expand, s])  # (32, 32, 69, 14) ~ (32, 14) -> (32, 32, 69, 14)
        x_gtu = []
        x_gtu.append(self.gtu1(obs_seq_reweighted))  # B,F,N,T
        x_gtu.append(self.gtu3(obs_seq_reweighted))  # B,F,N,T-2
        x_gtu.append(self.gtu5(obs_seq_reweighted))  # B,F,N,T-4
        x_gtu.append(self.gtu7(obs_seq_reweighted))  # B,F,N,T-6
        time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,4T-12
        time_conv = self.final_fc(time_conv)
        time_conv = self.obs_end_conv(time_conv)
        output = torch.squeeze(time_conv)
        return output, hidden

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(self.lstm_num_layers, batch_size * self.n_nodes, self.lstm_hidden_dim),
                  weight.new_zeros(self.lstm_num_layers, batch_size * self.n_nodes,
                                   self.lstm_hidden_dim))  # tuple, ((3, 1280, 64), (3, 1280, 64))
        return hidden


class PE_MDGCN(nn.Module):
    def __init__(self, adj_len: int, seq_len: int, n_nodes: int, input_dim: int, lstm_hidden_dim: int,
                 lstm_num_layers: int,
                 gcn_hidden_dim: int, sta_kernel_config: dict, gconv_use_bias: bool, gconv_activation=nn.ReLU,
                 sta_adj_list=None, device='cpu',
                 days=None, dims=None, num_nodes=None, dataset='taxi', output_len=1):
        super().__init__()
        self.M = adj_len
        self.sta_K = self.get_support_K(sta_kernel_config)  # 3
        self.rnn_list, self.gcn_list = nn.ModuleList(), nn.ModuleList()
        self.E_list = []
        for m in range(self.M):
            SFA_ST_Block = ModuleBlock(seq_len=seq_len, n_nodes=n_nodes, input_dim=input_dim,
                                       lstm_hidden_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers,
                                       K=self.sta_K, gconv_use_bias=gconv_use_bias, gconv_activation=gconv_activation,
                                       device=device, days=days, dims=dims, num_nodes=num_nodes, is_dynamic=False)
            self.rnn_list.append(SFA_ST_Block)
            gcn = GCN(K=self.sta_K, input_dim=lstm_hidden_dim, hidden_dim=gcn_hidden_dim,  # 3 64 64
                      bias=gconv_use_bias, activation=gconv_activation, weight_dim=n_nodes)
            self.gcn_list.append(gcn)
            self.E_list.append(nn.Parameter(torch.randn(num_nodes, 8), requires_grad=True).to(device))
        self.PAL_ST_Block = ModuleBlock(seq_len=seq_len, n_nodes=n_nodes, input_dim=input_dim,
                                        lstm_hidden_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers,
                                        K=self.sta_K, gconv_use_bias=gconv_use_bias, gconv_activation=gconv_activation,
                                        device=device, days=days, dims=dims, num_nodes=num_nodes, is_dynamic=True
                                        )
        self.dynamic_gcn = New_GCN(c_in=32, c_out=32, dropout=0.3)
        self.fc = nn.Linear(in_features=gcn_hidden_dim, out_features=1, bias=True)  # 64 1  64 output_len
        self.sta_adj_list = sta_adj_list

        # Period Dynamic Arrival Learning module


        if dataset == 'taxi':
            self.static_arrival = np.load('./data/complete/nyc_taxi_history_matrix_3d_69_5-8.npy', allow_pickle=True)
        else:
            self.static_arrival = np.load('./data/complete/nyc_bike_history_matrix_3d_104_5-8.npy', allow_pickle=True)
        self.static_arrival = torch.tensor(self.static_arrival, requires_grad=False).float().to(device)
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=32,
                                    kernel_size=(1, 1))
        self.end_conv = nn.Conv2d(in_channels=32,
                                  out_channels=1,
                                  kernel_size=(1, 1),
                                  bias=True)
        self.batch_norm = nn.BatchNorm2d(32)

    @staticmethod
    def PDAL(time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    @staticmethod
    def get_support_K(config: dict):
        if config['kernel_type'] == 'localpool':
            assert config['K'] == 1
            K = 1
        elif config['kernel_type'] == 'chebyshev':
            K = config['K'] + 1
        elif config['kernel_type'] == 'random_walk_diffusion':
            K = config['K'] * 2 + 1
        else:
            raise ValueError('Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion].')
        return K

    def init_hidden_list(self, batch_size: int):
        hidden_list = list()
        for m in range(self.M):
            hidden = self.rnn_list[m].init_hidden(batch_size)
            hidden_list.append(hidden)
        # 加了一个动态图
        hidden_list.append(self.PAL_ST_Block.init_hidden(batch_size))
        return hidden_list

    def forward(self, obs_seq: torch.Tensor, idn):
        # Static Feature Dynamic Adaptation
        assert len(self.sta_adj_list) == self.M
        batch_size = obs_seq.shape[0]
        hidden_list = self.init_hidden_list(
            batch_size)  # list:2 [((3, 1280, 64), (3, 1280, 64)), ((3, 1280, 64), (3, 1280, 64))]
        feat_list = list()
        for m in range(self.M):
            # SFDA
            cg_rnn_out, hidden_list[m] = self.rnn_list[m](self.sta_adj_list[m], obs_seq, hidden_list[m])  # (32, 40, 64)
            gcn_out = self.gcn_list[m](self.sta_adj_list[m], cg_rnn_out, conv_E=self.E_list[m])
            feat_list.append(gcn_out)  # [(32, 40, 64), ...]

        # Period Dynamic Arrival Graph Constructing
        adp = self.PDAL(self.nodevec_p1[idn], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        adp = self.static_arrival[idn] + adp
        cg_rnn_out, hidden_list[-1] = self.PAL_ST_Block(adp, obs_seq, hidden_list[-1], is_dynamic=True)

        tmp_x_seq = torch.unsqueeze(cg_rnn_out, dim=1)
        tmp_x_seq = self.start_conv(tmp_x_seq)
        gcn_out = self.dynamic_gcn(tmp_x_seq, [adp])
        x_seq_gconv = self.end_conv(gcn_out)
        x_seq_gconv = torch.squeeze(x_seq_gconv)
        feat_list.append(x_seq_gconv)
        feat_fusion = torch.sum(torch.stack(feat_list, dim=-1), dim=-1)  # aggregation
        output = self.fc(feat_fusion)
        return output
