import sys 
sys.path.append("..")
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
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

import sklearn.cluster as cluster
from sklearn.metrics import roc_auc_score


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.utils import Grad
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from .utils import load_txt, load_video2path_dict, load_pkl, save_json

def parse_args():
    parser = argparse.ArgumentParser(description='get embeddings for different model on contrast clip pairs')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--data_root', default='/data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity/timesformer/val_256_mask')
    parser.add_argument('--model_name', default='timesformer', help='save binary classifier weights')
    parser.add_argument('--target_layer', default=-1, type=int, help='save binary classifier weights')
    
    args = parser.parse_args()
    return args




def get_concept_vector_embeddings_mean(emb_dict, target_layer, device, model_name='timesformer'):
    idx2concept = []
    all_concept_vector = []
    if model_name == 'slowfast':
        for concept in emb_dict.keys():
            concept_matrix = []
            for video_name in emb_dict[concept].keys():
                #print('{} {} {}'.format(concept, video_name, type(emb_dict[concept][video_name][target_layer])))
                emb_slow = emb_dict[concept][video_name]['backbone.slow_path.layer4.2']
                emb_fast = emb_dict[concept][video_name]['backbone.fast_path.layer4.2']
                concept_matrix.append(F.normalize(torch.cat([emb_slow, emb_fast], dim=-1), dim=-1))
                #concept_matrix.append(emb_dict[concept][video_name][target_layer])
            concept_matrix = torch.stack(concept_matrix, dim=0)
            all_concept_vector.append(concept_matrix.mean(0))
            idx2concept.append(concept)
    else:
        for concept in emb_dict.keys():
            concept_matrix = []
            for video_name in emb_dict[concept].keys():
                #print('{} {} {}'.format(concept, video_name, type(emb_dict[concept][video_name][target_layer])))
                concept_matrix.append(F.normalize(emb_dict[concept][video_name][target_layer], dim=-1))
                #concept_matrix.append(emb_dict[concept][video_name][target_layer])
            concept_matrix = torch.stack(concept_matrix, dim=0)
            all_concept_vector.append(concept_matrix.mean(0))
            idx2concept.append(concept)
    if len(all_concept_vector[0].size()) == 2:
        all_concept_vector = torch.cat(all_concept_vector, dim=0)
    else:
        all_concept_vector = torch.stack(all_concept_vector, dim=0)
    idx2concept = dict(zip(range(len(idx2concept)), idx2concept))
    return idx2concept, all_concept_vector, None

def get_concept_vector_embeddings_all(emb_dict, target_layer, device, model_name='timesformer'):
    idx2concept = []
    all_concept_vector = []
    if model_name == 'slowfast':
        for concept in emb_dict.keys():
            for video_name in emb_dict[concept].keys():
                #print('{} {} {}'.format(concept, video_name, type(emb_dict[concept][video_name][target_layer])))
                #emb_slow = emb_dict[concept][video_name]['backbone.slow_path.layer4.2']
                #emb_fast = emb_dict[concept][video_name]['backbone.fast_path.layer4.2']
                #all_concept_vector.append(F.normalize(torch.cat([emb_slow, emb_fast], dim=-1), dim=-1))
                idx2concept.append('{}->{}'.format(concept, video_name))
                if target_layer in emb_dict[concept][video_name].keys():
                    emb_slow = emb_dict[concept][video_name][target_layer]
                    emb_fast = emb_dict[concept][video_name][target_layer.replace('slow_path', 'fast_path')]
                    all_concept_vector.append(F.normalize(torch.cat([emb_slow, emb_fast], dim=-1), dim=-1))
                else:
                    emb_slow = emb_dict[concept][video_name]['module.{}'.format(target_layer)]
                    emb_fast = emb_dict[concept][video_name]['module.{}'.format(target_layer).replace('slow_path', 'fast_path')]
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
    print(concept_matrix.size())
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
            algo = cluster.KMeans(n_clusters=n_clusters_concept, n_init=10, random_state=random_state)
        elif method == 'spectral':
            algo = cluster.SpectralClustering(n_clusters=n_clusters_concept, assign_labels='discretize', random_state=random_state)
        n_clusters = min(n_clusters, len(concept_matrix))
        print('process concept {}, {}'.format(concept_name, n_clusters_concept))
        results = algo.fit(vecs)
        
        labels = results.labels_
        try:
            centers = results.cluster_centers_
        except:
            print('no attribute cluster_centers_')
            centers = torch.zeros([10])
        centers_num = centers.shape[0]
        for label, video_name in zip(labels, concept_videoname_dict[concept_name]):
            concept_videoname_dict['{}->{}'.format(concept_name, label)].append(video_name)
        
        centers_matrix.append(centers)
        idx2center += ['{}->{}'.format(concept_name, label) for label in range(centers_num)]
    centers_matrix = np.concatenate(centers_matrix, axis=0)
    return idx2center, torch.Tensor(centers_matrix), concept_videoname_dict


def get_concept_score(grad, concept_matrix):
    concept_score = concept_matrix.matmul(grad)
    return concept_score

if __name__=='__main__':
    args = parse_args()
    data_root = args.data_root
    device = 'cuda:0'
    output_module = get_model_layers(args.model_name)
    video2path = load_video2path_dict('/data/shufan/shufan/mmaction2/data/kinetics400/vallist.txt')
    emb_dict = {}
    for pkl in os.listdir(data_root):
        cls_p = pkl.split('_')[0]
        emb_dict[cls_p] = load_pkl(os.path.join(data_root, pkl))
    #idx2concept, concept_matrix = get_concept_vector_weights(args.model_dir, output_module, device)
    idx2concept, concept_matrix, concept_videoname_dict = get_concept_vector_embeddings_cluster(emb_dict, output_module[args.target_layer], device)
'''    model = init_recognizer(args.config, args.checkpoint, device=device)
    #video_path = '/data1/shufan/mmaction2/data/kinetics400/val_256/blowing_out_candles/R-3sEFUy18Y.mp4'
    video_path = '/data1/shufan/mmaction2/data/kinetics400/val_256/dribbling_basketball/YXUsljH4iz8.mp4'    
    k = 20
    grad_getter = Grad(model, model.cfg, output_module)
    inputs = build_inputs(model, video_path, use_frames=False)
    grad, acti, label, _ = grad_getter(inputs)
    grad = F.normalize(grad[output_module[args.target_layer]][:, 0].mean(0), dim=0)

    concept_score = get_concept_score(grad.detach().cpu(), concept_matrix)
    values, indices = concept_score.topk(k, dim=0, largest=True, sorted=True)
    values, indices = values.detach().cpu(), indices.detach().cpu()
    print('largest:')
    for val, idx in zip(values, indices):
        #if '->' in idx2concept[int(idx)]:
        #    video_name = idx2concept[int(idx)].split('->')[1].split('.')[0]
        #    video_path = video2path[video_name]
        #    shutil.copyfile(video_path, osp.join('/data1/shufan/mmaction2/tools/clip_inference/demo/pos', osp.basename(video_path)))
        print('concept {}, score {}'.format(idx2concept[int(idx)], val))
        #print(concept_videoname_dict[idx2concept[int(idx)]][:5])
    print('smallest:')
    values, indices = concept_score.topk(k, dim=0, largest=False, sorted=True)
    values, indices = values.detach().cpu(), indices.detach().cpu()
    for val, idx in zip(values, indices):
        #if '->' in idx2concept[int(idx)]:
        #    video_name = idx2concept[int(idx)].split('->')[1].split('.')[0]
        #    video_path = video2path[video_name]
        #    shutil.copyfile(video_path, osp.join('/data1/shufan/mmaction2/tools/clip_inference/demo/neg', osp.basename(video_path)))
        print('concept {}, score {}'.format(idx2concept[int(idx)], val))
        #print(concept_videoname_dict[idx2concept[int(idx)]][:5])

    #cluster_name = 'person->2'
    #print(concept_videoname_dict[cluster_name][:5])
    #if not osp.exists(osp.join('/data1/shufan/mmaction2/tools/clip_inference/demo', cluster_name)):
    #    os.mkdir(osp.join('/data1/shufan/mmaction2/tools/clip_inference/demo', cluster_name))
    #for item in concept_videoname_dict[cluster_name][:5]:
    #    video_name = item.split('.')[0]
    #    video_path = video2path[video_name]
    #    shutil.copyfile(video_path, osp.join('/data1/shufan/mmaction2/tools/clip_inference/demo', cluster_name, osp.basename(video_path)))
#
    #cluster_name = 'person->3'
    #print(concept_videoname_dict[cluster_name][:5])
    #if not osp.exists(osp.join('/data1/shufan/mmaction2/tools/clip_inference/demo', cluster_name)):
    #    os.mkdir(osp.join('/data1/shufan/mmaction2/tools/clip_inference/demo', cluster_name))
    #for item in concept_videoname_dict[cluster_name][:5]:
    #    video_name = item.split('.')[0]
    #    video_path = video2path[video_name]
    #    shutil.copyfile(video_path, osp.join('/data1/shufan/mmaction2/tools/clip_inference/demo', cluster_name, osp.basename(video_path)))
'''