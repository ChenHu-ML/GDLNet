from utils.functions import trainNetwork, testNetwork
from GmSa.GSMA import GmAtt_mamem
from utils.GetMamem import getAllDataloader
import argparse
import torch

p_num = {
    1: 8,
    2: 8,
    3: 11,
    4: 7,
    5: 7,
    6: 8,
    7: 8,
    8: 11,
    9: 9,
    10: 7,
    11: 7
}

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=int, default=1, help='subject xx you want to train')
    ap.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    ap.add_argument('--wd', type=float, default=5e-2, help='weight decay')
    ap.add_argument('--iterations', type=int, default=150, help='number of training iterations')
    ap.add_argument('--bs', type=int, default=64, help='batch size')
    ap.add_argument('--model_path', type=str, default='../checkpoint/mamem/', help='saving the model path')
    ap.add_argument('--data_path', type=str, default='../data/MAMEM/', help='data path')
    ap.add_argument('--seed', type=int, default=3407, help='Specify the random seed for reproducibility')
    args = vars(ap.parse_args())
    
    print(f'subject{args["sub"]}')
    
    trainloader, validloader, testloader = getAllDataloader(subject=args['sub'], data_path=args['data_path'],
                                                            bs=args['bs'])

    net = GmAtt_mamem(p_num[args['sub']]).cpu()
    args.pop('bs')
    args.pop('data_path')
    net = trainNetwork(net, trainloader, validloader, testloader, **args)
    acc = testNetwork(net, testloader)
    print(f'{acc * 100:.2f}')
