from utils.functions import trainNetwork, testNetwork_auc
from GmSa.GSMA import GmAtt_cha
from utils.GetMamem import getAllDataloader
import argparse
import torch

p_num = {
    2: 3,
    6: 3,
    7: 7,
    11: 4,
    12: 6,
    13: 8,
    14: 3,
    16: 5,
    17: 5,
    18: 4,
    20: 5,
    21: 5,
    22: 3,
    23: 4,
    24: 6,
    26: 4
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=2, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=int, default=2, help='subjectxx you want to train')
    ap.add_argument('--lr', type=float, default=4e-2, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    ap.add_argument('--iterations', type=int, default=130, help='number of training iterations')
    ap.add_argument('--epochs', type=int, default=3,
                    help='number of epochs that you want to use for split EEG signals')
    ap.add_argument('--bs', type=int, default=30, help='batch size')
    ap.add_argument('--model_path', type=str, default='../checkpoint/BCIcha/',
                    help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default='../data/BCIcha/', help='data path')
    ap.add_argument('--plot_path', type=str, default='../plot/BCIcha/', help='plot path')
    args = vars(ap.parse_args())
    print(f'subject{args["sub"]}')
    trainLoader, validLoader, testLoader = getAllDataloader(subject=args['sub'], data_path=args['data_path'],
                                                            bs=args['bs'])

    args.pop('bs')
    args.pop('data_path')

    net = GmAtt_cha(p_num[args['sub']]).cpu()

    net = trainNetwork(net, trainLoader, validLoader, testLoader, **args)
    auc = testNetwork_auc(net, testLoader)
    print(f'{auc * 100:.2f}')
