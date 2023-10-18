import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from data.random_dataloader import RandomDataLoader


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class DataInput(object):
    def __init__(self, M_adj:int, data_dir:str, norm_opt=True, dataset='taxi'):
        self.M_sta = M_adj  # 图的个数
        self.data_dir = data_dir
        self.norm_opt = norm_opt
        self.data_scaler = MinMaxScaler()
        self.data_shape = None
        self.dataset = dataset

    def load_data(self):
        print('Loading data...')
        npy_data = np.load(self.data_dir, allow_pickle=True)
        self.data_shapeda = npy_data.shape
        dataset = dict()
        dataset['taxi'] = npy_data
        if self.dataset == 'taxi':
            if self.M_sta >= 1:
                dataset['neighbor_adj'] = np.load('./data/complete/nyc_taxi_dis_matrix_69_5-8.npy', allow_pickle=True)
            if self.M_sta >= 2:
                dataset['trans_adj'] = np.load('./data/complete/nyc_taxi_conn_matrix_69_5-8.npy', allow_pickle=True)
            if self.M_sta >= 3:
                dataset['semantic_adj'] = np.load('./data/complete/nyc_taxi_poi_matrix_69_5-8.npy', allow_pickle=True)
                pass
        elif self.dataset == 'bike':
            if self.M_sta >= 1:
                dataset['neighbor_adj'] = np.load('./data/complete/nyc_bike_dis_matrix_104_5-8.npy', allow_pickle=True)
            if self.M_sta >= 2:
                dataset['semantic_adj'] = np.load('./data/complete/nyc_bike_poi_matrix_104_5-8.npy', allow_pickle=True)
        return dataset

    def minmax_normalize(self, x:np.array):
        self._max, self._min = x.max(), x.min()
        print('min:', self._min, 'max:', self._max)
        x = (x - self._min) / (self._max - self._min)
        x = 2 * x - 1
        return x

    def minmax_denormalize(self, x:np.array):
        x = (x + 1)/2
        x = (self._max - self._min) * x + self._min
        return x

    def std_normalize(self, x:np.array):
        self._mean, self._std = x.mean(), x.std()
        print('mean:', round(self._mean, 4), 'std:', round(self._std, 4))
        x = (x - self._mean)/self._std
        return x

    def std_denormalize(self, x:np.array):
        x = x * self._std + self._mean
        return x


class TaxiDataset(Dataset):

    def __init__(self, device:str, inputs:dict, output:np.array, mode:str, mode_len:dict, start_idx:int, output_seq=1):
        self.device = device
        self.mode = mode
        self.mode_len = mode_len
        self.start_idx = start_idx  # train_start idx
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_seq'][item], self.output[item]

    def prepare_xy(self, inputs:dict, output:np.array):
        if self.mode == 'train':
            pass
        elif self.mode == 'validate':
            self.start_idx += self.mode_len['train']
        else:       # test
            self.start_idx += self.mode_len['train'] + self.mode_len['validate']

        obs = []
        for kw in ['weekly', 'daily', 'serial']:
            if len(inputs[kw].shape) != 2:      # dim=2 for empty seq
                obs.append(inputs[kw])
        x_seq = np.concatenate(obs, axis=1)   # concatenate timeslices to one seq
        x = dict()
        x['x_seq'] = torch.from_numpy(x_seq[self.start_idx: (self.start_idx + self.mode_len[self.mode])]).float().to(self.device)
        y = torch.from_numpy(output[self.start_idx: self.start_idx + self.mode_len[self.mode]]).float().to(self.device)
        return x, y



class DataGenerator(object):
    def __init__(self, dt, obs_len:tuple, train_test_dates:list, val_ratio:float, year=2022):
        self.days = int(24 // dt)  # dt 是指一小时为单位的时间步长
        self.serial_len, self.daily_len, self.weekly_len = obs_len  # 3 1 1 意思是考虑前三个时间步，考虑前一天的时间步，考虑前一周的时间步
        self.train_test_dates = train_test_dates        # [train_start, train_end, test_start, test_end]
        self.val_ratio = val_ratio
        self.start_idx, self.mode_len = self.date2len(year=year)

    def date2len(self, year:int):
        date_range = pd.date_range(str(year)+'0501', str(year)+'0830').strftime('%Y%m%d').tolist()
        train_s_idx, train_e_idx = date_range.index(str(year)+self.train_test_dates[0]),\
                                   date_range.index(str(year)+self.train_test_dates[1])
        train_len = (train_e_idx + 1 - train_s_idx) * self.days
        validate_len = int(train_len * self.val_ratio)
        train_len -= validate_len
        test_s_idx, test_e_idx = date_range.index(str(year)+self.train_test_dates[2]),\
                                 date_range.index(str(year)+self.train_test_dates[3])
        test_len = (test_e_idx + 1 - test_s_idx) * self.days
        return train_s_idx, {'train':train_len, 'validate':validate_len, 'test':test_len}

    def get_data_loader(self, data:dict, batch_size:int, device:str,
                        valid_batch_size=64, test_batch_size=64,
                        output_seq=1):
        feat_dict = dict()
        feat_dict['serial'], feat_dict['daily'], feat_dict['weekly'], output = self.get_feats(data['taxi'], output_seq)
        for k in feat_dict.keys():
            feat_dict[k] = np.expand_dims(feat_dict[k], 3)
        output = np.expand_dims(output, 2)

        data_loader = dict()        # data_loader for [train, validate, test]
        _len = output.shape[0]
        train_len = int(_len * 0.6)
        val_len = int(_len * 0.2)
        test_len = _len - train_len - val_len
        mode_len = {'train': train_len, 'validate': val_len, 'test': test_len}
        print(f'DataGenerator... Origin split: {feat_dict}')
        print(f'DataGenerator... Recalculate lens: train:[{train_len}], val:[{val_len}], test:[{test_len}]')


        for mode in ['train', 'validate', 'test']:
            dataset = TaxiDataset(device=device, inputs=feat_dict, output=output,
                                  mode=mode, mode_len=mode_len, start_idx=self.start_idx, output_seq=1)
            # data_loader[mode] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            data_loader[f'x_{mode}'] = dataset.inputs['x_seq']
            tmp_shape = dataset.output.shape
            data_loader[f'y_{mode}'] = np.array(dataset.output.cpu()).reshape((-1, tmp_shape[1], tmp_shape[3], tmp_shape[2]))
            if mode == "train":
                data_loader['scaler'] = StandardScaler(mean=dataset.inputs['x_seq'][..., 0].mean(), std=dataset.inputs['x_seq'][..., 0].std())
        for si in range(0, data_loader['x_' + mode].shape[-1]):  # 对每个特征的归一化，但是这里只有一个
            scaler_tmp = StandardScaler(mean=data_loader['x_train'][..., si].mean(), std=data_loader['x_train'][..., si].std())
            for category in ['train', 'validate', 'test']:
                data_loader['x_' + category][..., si] = scaler_tmp.transform(data_loader['x_' + category][..., si])
        data_loader['train_loader'] = RandomDataLoader(data_loader['x_train'], data_loader['y_train'], batch_size, days=self.days,
                                                       begin=0)
        data_loader['val_loader'] = RandomDataLoader(data_loader['x_validate'], data_loader['y_validate'], valid_batch_size, days=self.days,
                                                     begin=data_loader['x_train'].shape[0])
        data_loader['test_loader'] = RandomDataLoader(data_loader['x_test'], data_loader['y_test'], test_batch_size, days=self.days,
                                                      begin=data_loader['x_train'].shape[0] + data_loader['x_validate'].shape[0])

        return data_loader

    def get_feats(self, data:np.array, output_seq=1):
        serial, daily, weekly, y = [], [], [], []
        start_idx = max(self.serial_len, self.daily_len * self.days, self.weekly_len * self.days * 7)
        start_idx = int(start_idx)
        for i in range(start_idx, data.shape[0] - output_seq):
            serial.append(data[i - self.serial_len: i])  # 前三个时间步的
            daily.append(self.get_periodic_skip_seq(data, i, 'daily'))
            weekly.append(self.get_periodic_skip_seq(data, i, 'weekly'))
            y.append(data[i: i+output_seq])  # 就是用 serial daily weekly 去预测当前 y
        return np.array(serial), np.array(daily), np.array(weekly), np.array(y)

    def get_periodic_skip_seq(self, data:np.array, idx:int, p:str):
        p_seq = list()
        if p == 'daily':
            p_steps = self.daily_len * self.days  # 从最开始的前 n 周那一天开始
            p_steps = int(p_steps)
            for d in range(1, self.daily_len + 1):
                p_seq.append(data[idx - p_steps * d])
        else:   # weekly
            p_steps = self.weekly_len * self.days * 7
            p_steps = int(p_steps)
            for w in range(1, self.weekly_len + 1):
                p_seq.append(data[idx - p_steps * w])
        p_seq = p_seq[::-1]     # inverse order  放数据的时候是从远的开始放，现在反过来
        return np.array(p_seq)


