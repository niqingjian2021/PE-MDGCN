import torch
import torch.optim as optim
from torch import nn

import PEMDGCN
from util import TrainUtil


class trainer:

    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, normalization, lrate, wdecay, device,
                 days=288,
                 dims=40, order=2, sta_kernel_config=None,
                 sta_adj_list=None,
                 M=3, dataset='taxi'):
        self.model = PEMDGCN.PE_MDGCN(adj_len=M, seq_len=14, n_nodes=num_nodes, input_dim=1, lstm_hidden_dim=64,
                                      lstm_num_layers=3, gcn_hidden_dim=64,
                                      sta_kernel_config=sta_kernel_config,
                                      gconv_use_bias=True, gconv_activation=nn.ReLU,
                                      sta_adj_list=sta_adj_list,
                                      device=device,
                                      days=days,
                                      dims=dims,
                                      num_nodes=num_nodes,
                                      dataset=dataset,
                                      output_len=seq_length)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        self.loss = TrainUtil.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, ind):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input, ind)
        # output here is [32, 69, 1], switch it into [32, 1, 69, 1]
        output_shape = output.shape
        output = output.reshape(-1, 1, output_shape[1], output_shape[2])
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = TrainUtil.masked_mae(predict, real, 0.0).item()
        mape = TrainUtil.masked_mape(predict, real, 0.0).item()
        rmse = TrainUtil.masked_rmse(predict, real, 0.0).item()
        return mae, mape, rmse

    def eval(self, input, real_val, ind):
        self.model.eval()
        # input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input, ind)
        output_shape = output.shape
        output = output.reshape(-1, 1, output_shape[1], output_shape[2])
        output = output.transpose(1, 3)

        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        mae = TrainUtil.masked_mae(predict, real, 0.0).item()
        mape = TrainUtil.masked_mape(predict, real, 0.0).item()
        rmse = TrainUtil.masked_rmse(predict, real, 0.0).item()
        return mae, mape, rmse
