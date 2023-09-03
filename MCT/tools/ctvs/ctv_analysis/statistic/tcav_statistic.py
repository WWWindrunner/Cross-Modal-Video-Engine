import sys 
sys.path.append("..")
import os
import os.path as osp
import argparse
import random
from model_layers import get_model_layers

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import torch
import json
from tqdm import tqdm
from collections import defaultdict
from concept_score import get_concept_score, get_concept_vector_embeddings_mean, get_concept_vector_embeddings_all
from probing import TimeSformerHead, build_optimizer, top_k_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='get embeddings for different model on contrast clip pairs')
    parser.add_argument('--data_root', default='/data1/shufan/mmaction2/data/kinetics400')
    parser.add_argument('--grad_root', default='/data1/shufan/mmaction2/data/kinetics400')
    parser.add_argument('--save_dir', default='/data1/shufan/mmaction2/tools/clip_inference/tcav_statistics')
    parser.add_argument('--model_name', default='timesformer', help='save binary classifier weights')
#    parser.add_argument('--video_per_classes', default=20, type=int, help='save binary classifier weights')
    parser.add_argument('--target_layer', default=-1, type=int, help='save binary classifier weights')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    
    
    args = parser.parse_args()
    return args

def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    data = [line.strip().split(' ') for line in data]
    return data

def save_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(save_path, feats):
    with open(save_path, 'wb') as f:
        pickle.dump(feats, f)

class CBMHead(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes=1,
                 hidden_dims=2048,
                 init_std=0.02):
        super().__init__()
        self.init_std = init_std
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.cls_1 = nn.Linear(self.in_channels, self.hidden_dims)
        self.cls_2 = nn.Linear(self.hidden_dims, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.cls_1, std=self.init_std)
        trunc_normal_init(self.cls_2, std=self.init_std)

    def forward(self, x):
        # [N, in_channels]
        #x = torch.cat([emb_raw, emb_mask], dim=-1)
        x = self.cls_1(x)
        x = self.cls_2(x)
        # [N, num_classes]
        return x

def collate_data(args):
    data_root = args.grad_root
    matrix_dict = defaultdict(list)
    for pkl in os.listdir(data_root):
        pkl_path = os.path.join(data_root, pkl)
        pkl_data = load_pkl(pkl_path)
        if 'train' not in pkl:
            for layer in pkl_data.keys():
                matrix_dict[layer].append(pkl_data[layer])
    for layer in matrix_dict.keys():
        matrix_dict[layer] = torch.cat(matrix_dict[layer])
    
    return matrix_dict


class Cbm_Dataset(Dataset):
    def __init__(self, data, target_layer):
        self.data = []
        for vector, label in zip(data[target_layer], data['label']):
            self.data.append({'data': vector, 'label': label})
    def __getitem__(self, idx):
        label = self.data[idx]['label']
        data = self.data[idx]['data']
        return data, label
    def __len__(self):
        return len(self.data)





def tcav_statistic(idx2concept, dataloader, concept_matrix):
    concept_matrix = concept_matrix.cuda()
    result_dict = defaultdict(list)
    result_dict_concept = defaultdict(dict)
    sample_num_dict = defaultdict(int)
    concept_num_dict = defaultdict(int)
    for emb, label in tqdm(dataloader):
        emb = emb.cuda()
        concept_score = emb.matmul(concept_matrix.T)
        for c, lb in zip(concept_score, label):
            if result_dict[int(lb)] == []:
                result_dict[int(lb)] = F.normalize(c, dim=0)
            else:
                result_dict[int(lb)] += F.normalize(c, dim=0)
            sample_num_dict[int(lb)] += 1
    
    for idx in idx2concept.keys():
        concept_name = idx2concept[idx].split('->')[0]
        concept_num_dict[concept_name] += 1
    for lb in tqdm(result_dict.keys()):
        result_dict[int(lb)] /= sample_num_dict[int(lb)]
        for idx in range(len(result_dict[int(lb)])):
            concept_name = idx2concept[idx].split('->')[0]
            if float(result_dict[int(lb)][idx]) > 0:
                if concept_name in result_dict_concept[int(lb)].keys():
                    result_dict_concept[int(lb)][concept_name] += abs(float(result_dict[int(lb)][idx])) / concept_num_dict[concept_name]
                else:
                    result_dict_concept[int(lb)][concept_name] = abs(float(result_dict[int(lb)][idx])) / concept_num_dict[concept_name]

    return result_dict_concept

def tcav_statistic_mean(idx2concept, dataloader, concept_matrix):
    concept_matrix = concept_matrix.cuda()
    concept_matrix = F.normalize(concept_matrix, dim=-1)
    result_dict = defaultdict(list)
    result_dict_concept = defaultdict(dict)
    sample_num_dict = defaultdict(int)
    for emb, label in tqdm(dataloader):
        emb = emb.cuda()
        emb = F.normalize(emb, dim=-1)
        concept_score = emb.matmul(concept_matrix.T)
        for c, lb in zip(concept_score, label):
            if result_dict[int(lb)] == []:
                result_dict[int(lb)] = c
            else:
                result_dict[int(lb)] += c
            sample_num_dict[int(lb)] += 1
    
    
    for lb in tqdm(result_dict.keys()):
        result_dict[int(lb)] /= sample_num_dict[int(lb)]
        for idx in range(len(result_dict[int(lb)])):
            concept_name = idx2concept[idx]
            if concept_name in result_dict_concept[int(lb)].keys():
                result_dict_concept[int(lb)][concept_name] += float(result_dict[int(lb)][idx])
            else:
                result_dict_concept[int(lb)][concept_name] = float(result_dict[int(lb)][idx])

    return result_dict_concept

def tcav_statistic_fast(idx2concept, dataloader, concept_matrix):
    concept_matrix = concept_matrix.cuda()
#    concept_matrix = F.normalize(concept_matrix, dim=-1)
    result_dict = defaultdict(list)
    result_dict_concept = defaultdict(list)
    final_result_dict = defaultdict(dict)
    sample_num_dict = defaultdict(int)
    concept_num_dict = defaultdict(int)
    for emb, label in tqdm(dataloader):
        emb = emb.cuda()
#        emb = F.normalize(emb, dim=-1)
        concept_score = emb.matmul(concept_matrix.T)
        for c, lb in zip(concept_score, label):
            if result_dict[int(lb)] == []:
                c_norm = F.normalize(c, dim=0)
                #result_dict[int(lb)] = c_norm
                result_dict[int(lb)] = torch.max(c_norm, torch.zeros_like(c_norm))
            else:
                c_norm = F.normalize(c, dim=0)
                #result_dict[int(lb)] += c_norm
                result_dict[int(lb)] += torch.max(c_norm, torch.zeros_like(c_norm))
            sample_num_dict[int(lb)] += 1
    
    for idx in idx2concept.keys():
        concept_name = idx2concept[idx].split('->')[0]
        concept_num_dict[concept_name] += 1
    result_matrix = []
    for lb in tqdm(sorted(result_dict.keys())):
        result_dict[int(lb)] /= sample_num_dict[int(lb)]
        result_matrix.append(result_dict[int(lb)])
    result_matrix = torch.stack(result_matrix, dim=0).T
    print(result_matrix.size())
    for idx in tqdm(range(result_matrix.size()[0])):
        concept_name = idx2concept[idx]#.split('->')[0]
        #if concept_name in result_dict_concept.keys():
        #    result_dict_concept[concept_name] += result_matrix[idx] / concept_num_dict[concept_name]
        #else:
        #    result_dict_concept[concept_name] = result_matrix[idx]  / concept_num_dict[concept_name]
        result_dict_concept[concept_name] = result_matrix[idx]#  / concept_num_dict[concept_name]
    for concept_name, value in result_dict_concept.items():
        for lb in range(len(value)):
            final_result_dict[int(lb)][concept_name] = float(value[lb])
    return final_result_dict

if __name__=='__main__':
    args = parse_args()
    device = 'cuda:0'
    matrix_dict = collate_data(args)
    target_layer = 'module.{}'.format(get_model_layers(args.model_name)[args.target_layer])

    data_root = args.data_root
    emb_dict = {}
    for pkl in os.listdir(data_root):
        cls_p = pkl.split('_')[0]
        emb_dict[cls_p] = load_pkl(os.path.join(data_root, pkl))
    output_module = get_model_layers(args.model_name)
    #idx2concept, concept_matrix = get_concept_vector_weights(args.model_dir, output_module, device)
    idx2concept, concept_matrix, _ = get_concept_vector_embeddings_all(emb_dict, output_module[args.target_layer], device)

    dataset = Cbm_Dataset(matrix_dict, target_layer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    #result_dict_concept = tcav_statistic(idx2concept, dataloader, concept_matrix)
    result_dict_concept = tcav_statistic_fast(idx2concept, dataloader, concept_matrix)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, '{}_fast.json'.format(args.model_name))
    save_json(save_path, result_dict_concept)