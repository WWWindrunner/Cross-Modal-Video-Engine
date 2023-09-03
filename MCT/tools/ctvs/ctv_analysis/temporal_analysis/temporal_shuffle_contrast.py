import sys 
sys.path.append("..")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_layers import get_model_layers
from mmaction.apis import init_recognizer, inference_recognizer
from tqdm import tqdm
import pickle
import argparse
import torch.nn as nn
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import numpy as np
from audtorch.metrics.functional import pearsonr
from concept_score import get_concept_score, get_concept_vector_embeddings_mean, get_concept_vector_embeddings_all, get_concept_vector_embeddings_cluster
import json

def load_embdict(data_root, target_concept):
    emb_dict = {}
    for pkl in os.listdir(data_root):
        cls_p = pkl.split('_')[0]
        if cls_p != target_concept:
            continue
        print('loading {}'.format(cls_p))
        emb_dict[cls_p] = load_pkl(os.path.join(data_root, pkl))
        
    return emb_dict

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

def get_concept_vector_embeddings_all(emb_dict, target_layer, device, model_name='timesformer'):
    idx2concept = []
    all_concept_vector = []
    if model_name == 'slowfast':
        for concept in emb_dict.keys():
            print(concept)
            for video_name in emb_dict[concept].keys():
                #print('{} {} {}'.format(concept, video_name, type(emb_dict[concept][video_name][target_layer])))
                emb_slow = emb_dict[concept][video_name]['module.backbone.slow_path.layer4.2']
                emb_fast = emb_dict[concept][video_name]['module.backbone.fast_path.layer4.2']
                all_concept_vector.append(F.normalize(torch.cat([emb_slow, emb_fast], dim=-1), dim=-1))
                idx2concept.append('{}->{}'.format(concept, video_name))
    else:
        for concept in emb_dict.keys():
            for video_name in emb_dict[concept].keys():
                #print('{} {} {}'.format(concept, video_name, type(emb_dict[concept][video_name][target_layer])))
                if target_layer in emb_dict[concept][video_name].keys():
                    all_concept_vector.append(F.normalize(emb_dict[concept][video_name][target_layer], dim=-1))
                else:
                    all_concept_vector.append(F.normalize(emb_dict[concept][video_name]['module.{}'.format(target_layer)], dim=-1))
                idx2concept.append('{}->{}'.format(concept, video_name))
    if len(all_concept_vector[0].size()) == 2:
        all_concept_vector = torch.cat(all_concept_vector, dim=0)
    else:
        all_concept_vector = torch.stack(all_concept_vector, dim=0)
    idx2concept = dict(zip(range(len(idx2concept)), idx2concept))
    return idx2concept, all_concept_vector, None


def get_concept_vector_embeddings_cluster(emb_dict, target_layer, device, model_name='timesformer', method='kmeans', n_clusters=10, person_cluster=100, random_state=42, target_concept=None):
    idx2concept, concept_matrix, _ = get_concept_vector_embeddings_all(emb_dict, target_layer, None, model_name)
#    print(concept_matrix.size())
    idx2center, all_concept_vector, concept_videoname_dict = concept_cluster(idx2concept, concept_matrix, method=method, n_clusters=n_clusters, person_cluster=person_cluster, random_state=random_state, target_concept=target_concept)
    return idx2center, all_concept_vector, concept_videoname_dict

def concept_cluster(idx2concept, concept_matrix, method='kmeans', n_clusters=10, person_cluster=100, random_state=42, target_concept=None):
    concept_matrix_dict = defaultdict(list)
    concept_videoname_dict = defaultdict(list)
    for i, vec in enumerate(concept_matrix):
        concept_name = idx2concept[i].split('->')[0]
        concept_matrix_dict[concept_name].append(vec)
        concept_videoname_dict[concept_name].append(idx2concept[i].split('->')[1])
    for concept in concept_matrix_dict.keys():
        concept_matrix_dict[concept] = torch.stack(concept_matrix_dict[concept], dim=0)
    algo = None
    idx2center = []
    centers_matrix = []
    for concept_name, vecs in concept_matrix_dict.items():
        if target_concept is not None:
            if concept_name != target_concept:
                continue
        if concept_name == 'person':
            n_clusters_concept = min(person_cluster, len(vecs))
        else:
            n_clusters_concept = min(n_clusters, len(vecs))

        if method == 'kmeans':
            algo = cluster.KMeans(n_clusters=n_clusters_concept, init='k-means++', max_iter=300, random_state=random_state)
        n_clusters = min(n_clusters, len(concept_matrix))
        print('process concept {}, {}'.format(concept_name, n_clusters_concept))
        results = algo.fit(vecs)
        
        labels = results.labels_
        centers = results.cluster_centers_

        centers_num = centers.shape[0]
        for label, video_name in zip(labels, concept_videoname_dict[concept_name]):
            concept_videoname_dict['{}->{}'.format(concept_name, label)].append(video_name)
        
        centers_matrix.append(centers)
        idx2center += ['{}->{}'.format(concept_name, label) for label in range(centers_num)]
    centers_matrix = np.concatenate(centers_matrix, axis=0)
    return idx2center, torch.Tensor(centers_matrix), concept_videoname_dict


def get_temporal_proportion(target_concept, target_layer, emb_dict, shuffle_emb_dict, device, sample_num=10):
    result_dict = {}
#    max_cluster_num = len(emb_dict[target_concept].keys())
    max_cluster_num = 200
    interval = max_cluster_num // sample_num
    for n_cluster in range(1, max_cluster_num, interval):
        idx2concept, concept_matrix, concept_videoname_dict = get_concept_vector_embeddings_cluster(emb_dict, target_layer, device, model_name, n_clusters=n_cluster, person_cluster=n_cluster, target_concept=target_concept)
        n_cluster_result = []
        for cluster_idx in tqdm(range(n_cluster)):
            concept_name = '{}->{}'.format(target_concept, cluster_idx)
            concept_video_list = concept_videoname_dict[concept_name]
            
            concept_video_emb = []
            for video in concept_video_list:
                if target_layer in emb_dict[target_concept][video].keys():
                    print()
                    concept_video_emb.append(F.normalize(emb_dict[target_concept][video][target_layer], dim=-1))
                else:
                    try:
                        concept_video_emb.append(F.normalize(emb_dict[target_concept][video]['module.{}'.format(target_layer)], dim=-1))
                    except:
                        print(video, emb_dict[target_concept][video].keys())
            concept_video_emb = torch.stack(concept_video_emb, dim=0)

            shuffle_concept_video_emb = []
            for video in concept_video_list:
                if target_layer in shuffle_emb_dict[target_concept][video].keys():
                    shuffle_concept_video_emb.append(F.normalize(shuffle_emb_dict[target_concept][video][target_layer], dim=-1))
                else:
                    try:
                        shuffle_concept_video_emb.append(F.normalize(shuffle_emb_dict[target_concept][video]['module.{}'.format(target_layer)], dim=-1))
                    except:
                        print(video, shuffle_emb_dict[target_concept][video].keys())
        
            shuffle_concept_video_emb = torch.stack(shuffle_concept_video_emb, dim=0)
            
            center_emb = F.normalize(concept_video_emb.mean(0), dim=-1)
            shuffle_center_emb = F.normalize(shuffle_concept_video_emb.mean(0), dim=-1)
            
            corr_val = float(pearsonr(center_emb, shuffle_center_emb))
            n_cluster_result.append(corr_val)
        result_dict[n_cluster] = n_cluster_result
    return result_dict

if __name__=="__main__":
    #model_name = 'r2plus1d'
    model_name = 'slowfast'
    #model_name = 'timesformer'
    output_module = get_model_layers('{}_slow'.format(model_name))
    device = 'cuda:3'
    data_root = '/data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity_clip1/{}/val_256_mask'.format(model_name)
    shuffle_data_root = '/data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity_clip1/{}_shuffle'.format(model_name)
    #data_root = '/data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity/timesformer/val_256_mask'
    #shuffle_data_root = '/data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity/timesformer_shuffle'
    sample_num = 20
    emb_dict = load_embdict(data_root, target_concept='person')
    shuffle_emb_dict = load_embdict(shuffle_data_root, target_concept='person')
    
    #for layer_idx, target_layer in enumerate(output_module):
    target_layer = output_module[-1]
    
    result_dict = get_temporal_proportion(
        target_concept='person', 
        target_layer=target_layer, 
        emb_dict=emb_dict, 
        shuffle_emb_dict=shuffle_emb_dict, 
        device=device,
        sample_num=sample_num
    )
    #save_path = os.path.join('/data1/shufan/mmaction2/tools/clip_inference/temporal_shuffle', '{}_{}_clip1.json'.format(model_name, layer_idx))
    save_path = os.path.join('/data1/shufan/mmaction2/tools/clip_inference/temporal_shuffle', '{}_clip1.json'.format(model_name))
    save_json(save_path, result_dict)