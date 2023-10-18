import os
import argparse
import random
import time

import numpy as np
import torch
import Data_Container, GCN
from Model_Trainer import ModelTrainer
from util import TrainUtil
from util.trainer import trainer

data_dir = './data/complete/nyc_taxi_69_5-8.npy'
dt = 0.5      # time_slice
epoch = 1
batch_size = 32
learn_rate, weight_decay = 2e-3, 1e-4       # L2 regularization
M_adj = 3      # num static adjs
sta_kernel_config = {'kernel_type':'chebyshev', 'K':2}
loss_opt = 'MAE'
node_len = 69


def get_engines(args):
    global M_adj
    global data_dir
    device = torch.device(args.device)

    dataset = args.dataset
    if dataset == 'bike':
        M_adj = 2
        data_dir = './data/complete/nyc_bike_104_5-8.npy'

    data_in = Data_Container.DataInput(M_adj=M_adj, data_dir=data_dir, dataset=dataset)
    data = data_in.load_data()

    # prepare static adjs
    sta_adj_list = list()  # 静态图列表
    for key in list(data.keys()):
        if key.endswith('_adj'):
            adj_preprocessor = GCN.Adj_Preprocessor(**sta_kernel_config)
            adj = torch.from_numpy(data[key]).float()
            adj = adj_preprocessor.process(adj)
            sta_adj_list.append(adj.to(args.device))
    assert len(sta_adj_list) == M_adj     # ensure sta adj dim correct

    data_generator = Data_Container.DataGenerator(dt=dt, obs_len=obs_len, val_ratio=0.2, train_test_dates=dates)
    dataloader = data_generator.get_data_loader(data=data, batch_size=batch_size, device=args.device, output_seq=args.seq_length)
    scaler = dataloader['scaler']
    if dataset == 'taxi':
        num_nodes = 69
    else:
        num_nodes = 104

    engine = trainer(scaler, args.in_dim, args.seq_length, num_nodes, args.nhid, args.dropout, args.normalization,
                     args.learning_rate, args.weight_decay, device, days=args.days, dims=args.dims, order=args.order,
                     sta_kernel_config=sta_kernel_config, sta_adj_list=sta_adj_list, M=M_adj, dataset=dataset)
    return engine, dataloader


def main(args):
    engine, dataloader = get_engines(args)
    device = torch.device(args.device)
    scaler = dataloader['scaler']
    start_epoch = 1
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    count = 0

    for i in range(start_epoch, args.epochs + 1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        tt1 = time.time()
        dataloader['train_loader'].shuffle()
        for itera, (x, y, ind) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :], ind)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if itera % args.print_every == 0:
                # 调试专用！！！！！
                break
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(itera, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        tt2 = time.time()
        train_time.append(tt2 - tt1)
        # validate
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for itera, (x, y, ind) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :], ind)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        # early stopping
        if len(his_loss) > 0 and mvalid_loss < np.min(his_loss):
            count = 0
        else:
            count += 1
            print(f"no improve for {count} epochs")
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f},' \
              ' Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (tt2 - tt1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   os.path.join(args.save, "epoch_" + str(i) + "_" + str(round(float(mvalid_loss), 2)) + ".pth"))
        if count >= 50:
            break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # final test
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(
        os.path.join(args.save, "epoch_" + str(bestid + start_epoch)
                     + "_" + str(round(float(his_loss[int(bestid)]), 2)) + ".pth")))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for itera, (x, y, ind) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, ind)
            output_shape = preds.shape
            preds = preds.reshape(-1, 1, output_shape[1], output_shape[2])
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(float(his_loss[int(bestid)]), 4)))

    amae = []
    amape = []
    armse = []
    medae = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat)
        real = realy[:, :, i]
        metrics = TrainUtil.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d},' \
              ' Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

        print('-------------------- OLD --------------------')
        np_pred = np.array(pred.cpu())
        np_real = np.array(real.cpu())
        print('RMSE: ', ModelTrainer.RMSE(np_pred, np_real))
        print('MAE: ', ModelTrainer.MAE(np_pred, np_real))
        print('MAPE: ', ModelTrainer.MAPE(np_pred, np_real) * 100, '%')
        print('Masked MAPE: ', TrainUtil.masked_mape(pred,
                                                     real, 0.0).item())
        med = ModelTrainer.MedAE(np_pred, np_real)
        medae.append(med)
        print('-------------------- OLD --------------------')
        print('MedAE: ', ModelTrainer.MedAE(np_pred, np_real))

    log = 'On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f} medae: {:.3f}'
    print(log.format(args.seq_length, np.mean(amae), np.mean(amape), np.mean(armse), np.mean(medae)))
    torch.save(engine.model.state_dict(),
               os.path.join(args.save, "exp" + str(args.expid) +
                            "_best_" + str(round(float(his_loss[int(bestid)]), 2)) + ".pth"))
    return np.asarray(amae), np.asarray(amape), np.asarray(armse)


def only_test(args):
    engine, dataloader = get_engines(args)
    engine.model.load_state_dict(torch.load(
        args.save + '/' + 'v1.pth', map_location='cpu'))
    scaler = dataloader['scaler']
    device = torch.device(args.device)

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for itera, (x, y, ind) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, ind)
            output_shape = preds.shape
            preds = preds.reshape(-1, 1, output_shape[1], output_shape[2])
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    medae = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat)
        real = realy[:, :, i]
        metrics = TrainUtil.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d},' \
              ' Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])

        print('-------------------- OLD --------------------')
        np_pred = np.array(pred)
        np_real = np.array(real)
        print('RMSE: ', ModelTrainer.RMSE(np_pred, np_real))
        print('MAE: ', ModelTrainer.MAE(np_pred, np_real))
        print('MAPE: ', ModelTrainer.MAPE(np_pred, np_real) * 100, '%')
        print('Masked MAPE: ', TrainUtil.masked_mape(pred,
                                                     real, 0.0).item())
        med = ModelTrainer.MedAE(np_pred, np_real)
        print('MedAE: ', med)
        print('-------------------- OLD --------------------')
        amape.append(metrics[1])
        armse.append(metrics[2])
        medae.append(med)

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f} medae: {:.3f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(medae)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ST-MGCN')
    parser.add_argument('-device', '--device', type=str, help='Specify device usage',
                        choices=['cpu']+[f'cuda:{gpu}' for gpu in range(4)], default='cuda:2')  # todo change
    # parser.add_argument('-device', '--device', type=str, help='Specify device usage',
    #                     choices=['cpu']+[f'cuda:{gpu}' for gpu in range(4)], default='cpu')
    parser.add_argument('-model', '--model_name', type=str, help='Specify model_name',
                        choices=['STMGCN'], default='STMGCN')
    parser.add_argument('-date', '--dates', type=str, nargs='+',
                        help='Start/end dates of train/test sets. Test follows train.'
                             ' Example: -date 0101 0630 0701 0731',
                        default=['0501', '0731', '0801', '0830'])  # 0 ~ 1 是 train + val，2 ~ 3 是 test
    parser.add_argument('-cpt', '--obs_len', type=int, nargs='+',
                        help='Parameters for short-term/daily/weekly observations.'
                             ' Example: -cpt 3 1 1',
                        default=[12, 1, 1])  # 短期、天、周

    parser.add_argument('--seq_length', type=int, default=1, help='output length')
    parser.add_argument('--in_len', type=int, default=14, help='input length')
    parser.add_argument('--nhid', type=int, default=32, help='')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--print_every', type=int, default=50, help='')
    parser.add_argument('--runs', type=int, default=1, help='number of experiments')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument('--iden', type=str, default='', help='identity')
    parser.add_argument('--dims', type=int, default=32, help='dimension of embeddings for dynamic graph')
    parser.add_argument('--order', type=int, default=2, help='order of graph convolution')
    parser.add_argument('--num_nodes', type=int, default=69)
    parser.add_argument('--days', type=int, default=48)
    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dataset', type=str, default='taxi')
    args = parser.parse_args()

    seed_value = 3407
    if args.dataset == 'taxi':
        args.save = os.path.join('save_models/', 'nyc69' + args.iden)
    else:
        args.save = os.path.join('save_models/', 'bike104' + args.iden)
    os.makedirs(args.save, exist_ok=True)

    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

    torch.backends.cudnn.deterministic = True

    # parameters
    model_name = args.model_name
    dates = args.dates
    obs_len = tuple(args.obs_len)

    main(args)
