import sys 
sys.path.append("..")
import os
import os.path as osp
import shutil
import argparse
import random
import pickle
import numpy as np
import json
from tqdm import tqdm
from model_layers import get_model_layers
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import stats

from utils import load_txt, load_video2path_dict, load_pkl, save_json

def cal_m_v(emb_dict, keys_list, num_iters, num_samples):
    m_list, v_list, samples_list = [], [], []
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    for samples in tqdm(range(1, num_samples)):
        vector_mean_list = []
        for iters in range(num_iters):
            concept_matrix = []
            sample_list = random.sample(keys_list, samples)
            for video_name in sample_list:
                concept_matrix.append(F.normalize(emb_dict[concept][video_name][target_layer], dim=-1))
            concept_matrix = torch.stack(concept_matrix, dim=0)
            vector_mean_list.append(concept_matrix.mean(0))
        vector_mean_mat = torch.cat(vector_mean_list, dim=0)
        mean_vec = vector_mean_mat.mean(0)
        cos_dis_list = []
        for vec in vector_mean_mat:
            cos_dis_list.append(cos_sim(vec, mean_vec))
        cos_dis_list = np.array(cos_dis_list)
        m, v = np.mean(cos_dis_list), np.std(cos_dis_list)
        m_list.append(m)
        v_list.append(v)
        samples_list.append(samples)
    return m_list, v_list, samples_list

def cal_m_v_all(emb_dict, num_iters, num_samples, target_layer, model_type):
    m_list, v_list, samples_list = [], [], []
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    for samples in tqdm(range(1, num_samples, 50)):
        cos_dis_list = []
        for concept in emb_dict.keys():
            vector_mean_list = []
            keys_list = emb_dict[concept].keys()
            for iters in range(num_iters):
                concept_matrix = []
                sample_list = random.sample(keys_list, min(samples, len(keys_list)))
                for video_name in sample_list:
                    if target_layer not in set(emb_dict[concept][video_name].keys()):
                        concept_matrix.append(F.normalize(emb_dict[concept][video_name]['module.{}'.format(target_layer)], dim=-1))
                    else:
                        concept_matrix.append(F.normalize(emb_dict[concept][video_name][target_layer], dim=-1))
                if model_type == 'transformer':
                    concept_matrix = torch.stack(concept_matrix, dim=0)
                else:
                    concept_matrix = torch.stack(concept_matrix, dim=0).unsqueeze(1)
                vector_mean_list.append(concept_matrix.mean(0))
            vector_mean_mat = torch.cat(vector_mean_list, dim=0)
            mean_vec = vector_mean_mat.mean(0)
            for vec in vector_mean_mat:
                cos_dis_list.append(cos_sim(vec, mean_vec))
        cos_dis_list = np.array(cos_dis_list)
        m, v = np.mean(cos_dis_list), np.std(cos_dis_list)
        m_list.append(float(m))
        v_list.append(float(v))
        samples_list.append(samples)
    return m_list, v_list, samples_list


if __name__=="__main__":
    num_samples = 251
    num_iters = 100
    model_name_list = ['timesformer', 'r2plus1d', 'i3d', 'x3d', 'mvit', 'slowfast']
    model_type_list = ['transformer', '3DConv', '3DConv', '3DConv', '3DConv', '3DConv']
    #model_name_list = ['mvit', 'slowfast']
    #model_type_list = ['3DConv', '3DConv']
    m_dict, v_dict, samples_dict = {}, {}, {}
    save_dir = '/data1/shufan/mmaction2/tools/clip_inference/concept_vector_robustness'
    for model_name, model_type in zip(model_name_list, model_type_list):
        data_root = '/data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity/{}/val_256_mask'.format(model_name)
        output_module = get_model_layers(model_name)
        emb_dict = {}
        for pkl in os.listdir(data_root):
            cls_p = pkl.split('_')[0]
            print('loading {}...'.format(pkl))
            emb_dict[cls_p] = load_pkl(os.path.join(data_root, pkl))
        target_layer = output_module[-1]
        print('enumerate for model {}'.format(model_name))
        m_list, v_list, samples_list = cal_m_v_all(emb_dict, num_iters, num_samples, target_layer, model_type)
        m_dict[model_name] = m_list
        v_dict[model_name] = v_list
        samples_dict[model_name] = samples_list
    save_path = os.path.join(save_dir, 'm_dict.json')
    save_json(save_path, m_dict)
    save_path = os.path.join(save_dir, 'v_dict.json')
    save_json(save_path, v_dict)
    save_path = os.path.join(save_dir, 'samples_dict.json')
    save_json(save_path, samples_dict)
