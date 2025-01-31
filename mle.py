import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.DNN import DNN
from utils.data_process import ReadDataset
import torch.nn as nn
import numpy as np
from path_explain.utils import set_up_environment
from utils.path_explainer_torch import PathExplainerTorch
from utils.layers import FeaturesEmbedding
import matplotlib.pyplot as plt
from utils.functions import standardization, cluster_registration
import time
from utils.functions import model_train, model_test, tsne
import copy
from tqdm import tqdm

set_up_environment(visible_devices='0')


def main(model_name,
         dataset_name,
         source_name,
         data_dir,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         dropout,
         device,
         save_dir,
         mlp_dims,
         embed_dim,
         none_prob,
         sample_num):
    print("Model: ", model_name)
    print("Embed_dim: ", embed_dim)
    print("Dataset: ", dataset_name)
    print("Source: ", source_name)
    print("Data Dir: ", data_dir)
    print("Epoch: ", epoch)
    print("Learning Rate: ", learning_rate)
    print("Batch Size: ", batch_size)
    print("Weight_Decay: ", weight_decay)
    print("Mlp_dims: ", mlp_dims)
    print("Dropout: ", dropout)
    print("Device: ", device)
    print("Save Dir: ", save_dir)
    print("None Prob: ", none_prob)
    print("Sample Number: ", sample_num)

    print("------------------Preparing dataset and model...------------------")
    device = torch.device(device)
    train_set = ReadDataset(data_dir=data_dir, name=dataset_name, state='train', shuffle=True)
    field_dims = train_set.field_dims
    print("Field dims: ", field_dims)
    field_dims = field_dims[:-1]
    val_set = ReadDataset(data_dir=data_dir, name=dataset_name, state='val', shuffle=False)
    test_set = ReadDataset(data_dir=data_dir, name=dataset_name, state='test', shuffle=False)
    if source_name is not None:
        source_set = ReadDataset(data_dir=data_dir, name=dataset_name, state=source_name)
        # gjt, for sample num
        if sample_num is not None:
            source_set.field = source_set.field[:sample_num]
            source_set.label = source_set.label[:sample_num]
        print("Source : Train = ", len(source_set)/len(train_set))
        # train_set = source_set
        len_train = len(train_set)
        train_set.field = train_set.field[:int(0.1*len_train)]
        train_set.label = train_set.label[:int(0.1*len_train)]
        train_set = train_set.merge(source_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    print("Num of samples: Train: {}, Val: {}, Test: {}"
          .format(len(train_set), len(val_set), len(test_set)))

    if model_name == "dnn":
        model = DNN(field_dims, mlp_dims, dropout, emb_dim=embed_dim, dataset_name=dataset_name).to(device)


    print("----------Start training&testing...----------")
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if dataset_name in ["heloc", "ml-1m", "amazon", "douban"]:
        criterion = nn.BCELoss()
    elif dataset_name in ["covtype"]:
        criterion = nn.CrossEntropyLoss()
    elif dataset_name in ["housing"]:
        criterion = nn.MSELoss()
    start_time = time.time()
    result = model_train(model, train_loader, val_loader, optimizer, criterion,
                                       epoch, save_dir, device, test_loader=test_loader, dataset_name=dataset_name)
    end_time = time.time()
    print("Time Spent: {} second".format(end_time - start_time))
    return result



if __name__ == '__main__':
    import argparse
    model_list = ['llama3-70B_sampling', 'llama3-70B_random', 'ddpm', 'ctgan', 'adsgan', 'tvae', 'pategan', None]
    for source_name in model_list:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', default="dnn")
        parser.add_argument('--dataset_name', default="douban")
        # ddpm, ctgan, adsgan, tvae, pategan, llama3-70B_sampling
        parser.add_argument('--source_name', default=source_name)
        parser.add_argument('--data_dir', default='data')
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=500)
        parser.add_argument('--weight_decay', type=float, default=1e-6)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--embed_dim', type=int, default=16)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--mlp_dims', type=list, default=[128, 64, 32, 1])
        parser.add_argument(
            '--device', default='cuda:0')
        parser.add_argument('--none_prob', default=1e-10)
        parser.add_argument('--sample_num', default=None)
        parser.add_argument('--save_dir', default='log/model.pt')
        parser.add_argument('--repeat_experiments', type=int, default=5)
        args = parser.parse_args()
        result = []
        for i in range(args.repeat_experiments):
            result_tmp = main(args.model_name,
                     args.dataset_name,
                     args.source_name,
                     args.data_dir,
                     args.epoch,
                     args.learning_rate,
                     args.batch_size,
                     args.weight_decay,
                     args.dropout,
                     args.device,
                     args.save_dir,
                     args.mlp_dims,
                     args.embed_dim,
                     args.none_prob,
                     args.sample_num)
            result.append(list(result_tmp))
        if args.dataset_name in ["heloc", "ml-1m", "amazon", "douban"]:
            result = pd.DataFrame(result, columns=['AUC', 'Logloss'])
        elif args.dataset_name in ["covtype"]:
            result = pd.DataFrame(result, columns=['weighted-P', 'macro-P', 'micro-P', 'recall', 'f1'])
        result.to_csv("mle-result-{}-num{}-learn{}-{}.csv".format(args.dataset_name,args.sample_num, str(args.learning_rate), args.source_name))