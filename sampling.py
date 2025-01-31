import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model.DNN import DNN, Emb_Head
from utils.data_process import ReadDataset
import torch.nn as nn
import numpy as np
from path_explain.utils import set_up_environment
from utils.path_explainer_torch import PathExplainerTorch
from utils.layers import FeaturesEmbedding
import matplotlib.pyplot as plt
from utils.functions import standardization, cluster_registration, gather_nd
import time
from utils.functions import model_train, model_test, tsne
import copy
from tqdm import tqdm

set_up_environment(visible_devices='0')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 20


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
    field_dims = train_set.field_dims[:-1]
    val_set = ReadDataset(data_dir=data_dir, name=dataset_name, state='val', shuffle=False)
    test_set = ReadDataset(data_dir=data_dir, name=dataset_name, state='test', shuffle=False)
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
    start_time = time.time()
    _ = model_train(model, train_loader, val_loader, optimizer, criterion,
                               epoch, save_dir, device, test_loader=test_loader, dataset_name=dataset_name)
    end_time = time.time()
    print("Time Spent: {} second".format(end_time - start_time))

    print("------------------Clustering------------------")
    train_tensor = torch.from_numpy(train_set.field).to(device)
    train_label = torch.from_numpy(train_set.label).to(device)
    if embed_dim is not None:
        embedding_model = nn.Sequential()
        interpret_model = nn.Sequential()
        for layer in model.named_children():
            print("Name: ", layer[0])
            # print(layer[1])
            if isinstance(layer[1], FeaturesEmbedding):
                embedding_model.add_module(layer[0], layer[1])
            else:
                interpret_model.add_module(layer[0], layer[1])
        embedding_model.load_state_dict(torch.load(save_dir), strict=False)
        interpret_model.load_state_dict(torch.load(save_dir), strict=False)
        embedding_model.eval()
        interpret_model.eval()
        train_tensor = embedding_model(train_tensor).detach()
    else:
        interpret_model = copy.deepcopy(model)
        interpret_model.eval()

    sample_use = 10000 if len(train_tensor)>10000 else len(train_tensor)
    train_tensor = train_tensor[:sample_use]
    train_label = train_label[:sample_use]
    if dataset_name in ["heloc", "ml-1m", "amazon", "douban"]:
        train_label = train_label.float()
    else:
        train_label = train_label.long()
    if dataset_name in ["heloc", "ml-1m", "amazon", "douban"]:
        criterion = nn.BCELoss()
    elif dataset_name in ["covtype"]:
        criterion = nn.CrossEntropyLoss()
    explainer = PathExplainerTorch(interpret_model)
    baseline = torch.zeros_like(train_tensor)[:2]
    # print(train_tensor.shape)
    # train_tensor = interpret_model(train_tensor)
    # print(train_tensor.shape)
    train_tensor.require_grad = True

    # 第一种
    interactions = explainer.interactions(input_tensor=train_tensor,
                                          baseline=baseline,
                                          num_samples=36,
                                          use_expectation=True,
                                          output_indices=None,
                                          verbose=True,
                                          label=train_label,
                                          criterion=criterion)
    if embed_dim is not None:
        interactions_np = torch.mean(torch.sum(interactions.detach(), dim=-1), dim=0).cpu().numpy()
    else:
        interactions_np = torch.mean(interactions.detach(), dim=0).cpu().numpy()

    # no matter negative or positive
    interactions_np = abs(interactions_np)

    # gjt 判断是否特征交互正确
    # center is too big
    for i in range(interactions_np.shape[0]):
        interactions_np[i,i]=0

    # interactions_np = standardization(interactions_np.reshape(-1)).reshape(interactions_np.shape[0], interactions_np.shape[1])

    fea_fields = train_set.fea_fields
    # print(interactions_np)
    clusters, cluster_names = cluster_registration(interactions_np, 0.5, fea_fields)
    print("Clusters: ", clusters)


    # plt.xticks(np.arange(len(fea_fields)), labels=fea_fields,
    #            rotation=45, rotation_mode="anchor", ha="right")
    # plt.yticks(np.arange(len(fea_fields)), labels=fea_fields)

    plt.imshow(interactions_np)
    ticks = [i for i in range(0, len(interactions_np), 2)]
    ticks_new = [i + 1 for i in range(0, len(interactions_np), 2)]
    plt.xticks(ticks, ticks_new)
    plt.yticks(ticks, ticks_new)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("./fig/"+ dataset_name + '-inter.pdf')
    plt.show()

    print("------------------Calculating importance score------------------")
    source_set = ReadDataset(data_dir=data_dir, name=dataset_name, state=source_name)
    train_pd = train_set.to_pandas()
    sou_pd = source_set.to_pandas()
    tar_p = {}
    sou_p = {}
    # 组频次统计
    for idx, clu in enumerate(tqdm(cluster_names)):
        tar_p[idx] = train_pd.value_counts(subset=clu) / len(train_pd)
        sou_p[idx] = sou_pd.value_counts(subset=clu) / len(sou_pd)

    importance_score = np.ones(len(sou_pd))
    for idx, clu in enumerate(cluster_names):
        #source data for clu cluster
        data = list(sou_pd[clu].values)
        # source prob for clu cluster
        s_p = sou_p[idx].loc[data]
        t_p = []
        # for each cluster sample
        for dat in tqdm(data):
            # if this cluster sample exist in target dataset
            ext_dat = tar_p[idx].index.isin([tuple(dat)])
            ext_dat1 = ext_dat.any()
            # if yes, use tar_p for prob
            if ext_dat1:
                t_p.append(tar_p[idx].loc[tuple(dat)])
            # if not exist, prob is none_prob
            else:
                t_p.append(none_prob)
        # update importance score with this cluster
        importance_score = importance_score*np.array(t_p)/s_p.values


    train_set1, val_set1, test_set1 = source_set.split([0.8, 0.1])
    print("Num of samples: Train: {}, Val: {}, Test: {}"
          .format(len(train_set1), len(val_set1), len(test_set1)))
    train_loader1 = DataLoader(train_set1, batch_size=batch_size, shuffle=True)
    val_loader1 = DataLoader(val_set1, batch_size=batch_size, shuffle=False)
    test_loader1 = DataLoader(test_set1, batch_size=batch_size, shuffle=False)
    if model_name == "dnn":
        new_model = DNN(field_dims, mlp_dims, dropout, emb_dim=embed_dim, dataset_name=dataset_name).to(device)
    optimizer = torch.optim.Adam(
        params=new_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if dataset_name in ["heloc", "ml-1m", "amazon", "douban"]:
        criterion = nn.BCELoss()
    elif dataset_name in ["covtype"]:
        criterion = nn.CrossEntropyLoss()

    print("----------Training&testing for source model----------")
    time.sleep(0.1)
    start_time = time.time()
    _ = model_train(new_model, train_loader1, val_loader1, optimizer, criterion,
                               epoch, save_dir, device, test_loader1, dataset_name=dataset_name)
    end_time = time.time()
    print("Time Spent: {} second".format(end_time - start_time))

    ("------------------Refining importance score------------------")
    tar_all = []
    sou_all = []
    source_loader = DataLoader(source_set, batch_size=batch_size, shuffle=False)
    model.eval()
    new_model.eval()
    for field, label in source_loader:
        field, label = field.to(device), label.to(device)
        if embed_dim is None:
            field = field.float()
        else:
            field = field.long()
        # 想法: 每次每个label 均生成一个样本，根据xy取概率
        sample_indices = torch.arange(0, label.size(0)).to(device)
        indices_tensor = torch.cat([
            sample_indices.unsqueeze(1),
            label.unsqueeze(1)], dim=1)
        tar_pred = model(field)
        sou_pred = new_model(field)
        if len(tar_pred.shape)>1:
            tar_pred = torch.nn.functional.softmax(gather_nd(model(field), indices_tensor).squeeze(),dim=-1)
            sou_pred = torch.nn.functional.softmax(gather_nd(new_model(field), indices_tensor).squeeze(), dim=-1)
        tar_all.extend(tar_pred.detach().cpu().tolist())
        sou_all.extend(sou_pred.detach().cpu().tolist())
    # gjt 改这里
    tar_xy = np.array(tar_all)
    sou_xy = np.array(sou_all)
    importance_score = importance_score * tar_xy / sou_xy
    # norm the score
    importance_score = importance_score / importance_score.sum()
    # importance_score = 1-importance_score
    # importance_score = importance_score / importance_score.sum()
    # print(importance_score)
    # filter out samples with prob 0
    useful = np.where(importance_score > 0.0, 1, 0)
    print("Useful samples: ", useful.sum())

    print("------------------Sampling------------------")
    indexs = np.arange(len(source_set))
    sample1 = np.random.choice(indexs, size=sample_num, replace=False, p=useful / useful.sum())
    source_set1 = copy.deepcopy(source_set)
    source_set1.field = source_set1.field[sample1]
    source_set1.label = source_set1.label[sample1]
    sample2 = np.random.choice(indexs, size=sample_num, replace=False, p=importance_score)
    source_set2 = copy.deepcopy(source_set)
    source_set2.field = source_set2.field[sample2]
    source_set2.label = source_set2.label[sample2]

    print("------------------TSNE------------------")
    train_idx = np.random.choice(np.arange(len(train_set)), size=1000, replace=False)
    source1_idx = np.random.choice(np.arange(len(source_set1)), size=500, replace=False)
    source2_idx = np.random.choice(np.arange(len(source_set2)), size=500, replace=False)
    tsne(np.concatenate((train_set.field[train_idx], train_set.label[train_idx].reshape((-1, 1))), axis=1),
         np.concatenate((source_set1.field[source1_idx], source_set1.label[source1_idx].reshape((-1, 1))), axis=1))

    # tsne(np.concatenate((train_set.field[train_idx], train_set.label[train_idx].reshape((-1, 1))), axis=1),
    #      np.concatenate((source_set1.field[source1_idx], source_set1.label[source1_idx].reshape((-1, 1))), axis=1),
    #      np.concatenate((source_set2.field[source2_idx], source_set2.label[source2_idx].reshape((-1, 1))), axis=1))

    real_data = pd.read_csv("{}/{}/{}.csv".format(data_dir, dataset_name, dataset_name))
    sample_data = source_set1.to_pandas()
    sample_data.to_csv("{}/{}/{}_{}_random.csv".format(data_dir, dataset_name, dataset_name, source_name),
                          index=False)
    synthetic_data = source_set2.to_pandas()
    synthetic_data.to_csv("{}/{}/{}_{}_sampling.csv".format(data_dir, dataset_name, dataset_name, source_name), index=False)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    # quality_report = evaluate_quality(
    #     real_data,
    #     sample_data,
    #     metadata)
    #
    # quality_report = evaluate_quality(
    #     real_data,
    #     synthetic_data,
    #     metadata)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="dnn")
    parser.add_argument('--dataset_name', default="douban")
    parser.add_argument('--source_name', default='llama3-70B')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--mlp_dims', type=list, default=[128, 64, 32, 1])
    parser.add_argument(
        '--device', default='cuda:0')
    parser.add_argument('--none_prob', default=1e-20)
    parser.add_argument('--sample_num', default=2000)
    parser.add_argument('--save_dir', default='log/model.pt')
    args = parser.parse_args()
    main(args.model_name,
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
