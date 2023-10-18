import numpy as np
import pandas as pd

DIR = '../NYC_taxi/processed_data/'
FILE_NAME = 'data_dict.npz'

def generate():
    adj = pd.read_csv('../NYC_taxi/processed_data/adj.csv', header=None)
    simi = pd.read_csv('../NYC_taxi/processed_data/simi.csv', header=None)
    conn = pd.read_csv('../NYC_taxi/processed_data/conn.csv', header=None)
    obs = pd.read_csv('../NYC_taxi/processed_data/data_6-7_gap_10min.csv', index_col=0, header=None)
    np.savez(DIR + FILE_NAME,
             neighbor_adj=np.array(adj),
             trans_adj=np.array(conn),
             semantic_adj=np.array(simi),
             taxi=np.array(obs))


def seek():
    c = np.load(DIR + FILE_NAME, allow_pickle=True)
    for key, data in c.items():
        print(key, ": ", data)


if __name__ == '__main__':
    generate()
    seek()