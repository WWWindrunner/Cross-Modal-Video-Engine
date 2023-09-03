import sys 
sys.path.append("..")
import os
import os.path as osp
import random
import pickle
import numpy as np
import torch
import json
import argparse
from model_layers import get_model_layers
from collections import defaultdict
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f)

def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    data = [line.strip().split(' ') for line in data]
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='get embeddings for different model on contrast clip pairs')
    parser.add_argument('--data_root', default='/data/shufan/shufan/mmaction2/data/kinetics400/model_embeddings/entity/timesformer/train_256_mask')
    parser.add_argument('--test_size', type=float, default=0)
    parser.add_argument('--same_action', type=int, default=0)
    #parser.add_argument('--pos_contain_action', type=int, default=10)
    parser.add_argument('--pos_contain_action', default=40, type=int, nargs='+', help='action num contained in each concept positive samples')
    parser.add_argument('--neg_mask_action', default=40, type=int, nargs='+', help='action num masked in each concept negative samples')
    #parser.add_argument('--neg_mask_action', type=int, default=10)
    parser.add_argument('--save_dir', default='/data/shufan/shufan/mmaction2/tools/clip_inference/probing_split')
#    parser.add_argument('--save_contrast_path', default='/data1/shufan/mmaction2/tools/clip_inference/dataset_split.json')
    
    args = parser.parse_args()
    return args

def get_pos_list(emb_dict, class_name, concept_action_dict, concept_action_matrix, idx2class, concept2idx, contain_action=20):
    
    pos_concept_action_dist = concept_action_matrix.T[concept2idx[class_name]]
    legal_action = []
    sorted_dist, indices = torch.sort(pos_concept_action_dist, descending=True)
    for mask_num in range(contain_action):
        legal_action.append(idx2class[str(int(indices[mask_num]))])
    legal_video = []
    for action_name in concept_action_dict[class_name].keys():
        if action_name in legal_action:
            legal_video += concept_action_dict[class_name][action_name]
    legal_video = set(legal_video)
    pos_list = [video_name for video_name in emb_dict[class_name].keys() if video_name.split('.')[0] in legal_video]
    pos_contrast_list = [video_name for video_name in emb_dict[class_name].keys()]
    pos_contrast_list = random.sample(pos_contrast_list, len(pos_list))
    return pos_list, pos_contrast_list

def get_neg_list(emb_dict, class_name, pos_list, concept_action_dict, concept_action_matrix, idx2class, concept2idx, same_action=False, mask_action=20):
    neg_list = []
    neg_class_list = [key for key in emb_dict.keys() if class_name != key]
    num_samples_per_cls = max(len(pos_list) // len(neg_class_list), 1)
    
    pos_concept_action_dist = concept_action_matrix.T[concept2idx[class_name]]
    illegal_action = []
    # same_action is True means the action classes overlap increasing with mask_action
    if same_action:
        sorted_dist, indices = torch.sort(pos_concept_action_dist, descending=False)
        for mask_num in range(mask_action):
            illegal_action.append(idx2class[str(int(indices[mask_num]))])
    else:
        sorted_dist, indices = torch.sort(pos_concept_action_dist, descending=True)
        for mask_num in range(mask_action):
            illegal_action.append(idx2class[str(int(indices[mask_num]))])        
    for neg_cls in neg_class_list[:-1]:
        neg_cls_data = emb_dict[neg_cls]
        legal_video = []
        for action_name in concept_action_dict[neg_cls].keys():
            if action_name not in illegal_action:
                legal_video += concept_action_dict[neg_cls][action_name]
        legal_video = set(legal_video)
        
        neg_videos = ['{}->{}'.format(neg_cls, video_name) for video_name in neg_cls_data.keys() if video_name.split('.')[0] in legal_video]
        neg_list += random.sample(neg_videos, min(num_samples_per_cls, len(neg_videos)))
    neg_cls = neg_class_list[-1]
    neg_cls_data = emb_dict[neg_cls]
    neg_videos = ['{}->{}'.format(neg_cls, video_name) for video_name in neg_cls_data.keys() if video_name.split('.')[0] in legal_video]
    neg_list += random.sample(neg_videos, min(len(neg_videos), max(len(pos_list)-len(neg_list), 0)))
    return neg_list


def load_video2path_dict(file_path):
    data_root = os.path.dirname(file_path)
    data = load_txt(file_path)
    video2path = []
    for item in data:
        video_name = os.path.basename(item[0])
        video_name = video_name.split('.')[0]        
        video2path.append((video_name, os.path.join(data_root, item[0])))
    return dict(video2path)

def get_idx2class(file_path):
    ann_data = load_txt(file_path)
    idx2class = dict([(label, os.path.basename(os.path.dirname(path))) for path, label in ann_data])
    return idx2class
def get_class2idx(file_path):
    ann_data = load_txt(file_path)
    class2idx = dict([(os.path.basename(os.path.dirname(path)), label) for path, label in ann_data])
    return class2idx



def split_data(emb_dict, class_name, test_size=0.2, same_action=True, pos_contain_action=400, neg_mask_action=0):
    idx2class = get_idx2class('/data/shufan/shufan/mmaction2/data/kinetics400/trainlist.txt')
    class2idx = get_class2idx('/data/shufan/shufan/mmaction2/data/kinetics400/trainlist.txt')
    video2path = load_video2path_dict('/data/shufan/shufan/mmaction2/data/kinetics400/trainlist.txt')
    def action_dict():
        return defaultdict(list)
    

    concept_root = '/data/shufan/shufan/mmaction2/data/kinetics400/train_256_yolov5s_mask'
    concept_num = len(os.listdir(concept_root))
    action_num = len(class2idx)
    concept_action_matrix = torch.zeros((action_num, concept_num))
    concept2idx = {}
    idx2concept = {}

    for i, concept_name in enumerate(os.listdir(concept_root)):
        concept_path = os.path.join(concept_root, concept_name)
        concept2idx[concept_name] = i
        idx2concept[i] = concept_name
        for video_name in os.listdir(concept_path):
            video_path = video2path[video_name.split('.')[0]]
            label_str = os.path.basename(os.path.dirname(video_path))
            label_idx = int(class2idx[label_str])
            concept_action_matrix[label_idx][i] += 1

    concept_action_dict = defaultdict(action_dict)
    for concept_name in os.listdir(concept_root):
        concept_path = os.path.join(concept_root, concept_name)
        for video_name in os.listdir(concept_path):
            video_path = video2path[video_name.split('.')[0]]
            action_name = os.path.basename(os.path.dirname(video_path))
            concept_action_dict[concept_name][action_name].append(video_name.split('.')[0])


    pos_list, pos_contrast_list = get_pos_list(emb_dict, class_name, concept_action_dict, concept_action_matrix, idx2class, concept2idx, contain_action=pos_contain_action)
    neg_list = get_neg_list(emb_dict, class_name, pos_list, concept_action_dict, concept_action_matrix, idx2class, concept2idx, same_action=same_action, mask_action=neg_mask_action)
    
    pos_test_num = int(len(pos_list)*test_size)
    neg_test_num = int(len(neg_list)*test_size)
    print('{}, pos_num {}, pos_contrast_num {}, neg_num {}'.format(class_name, len(pos_list), len(pos_contrast_list), len(neg_list)))
    random.shuffle(pos_list)
    random.shuffle(neg_list)
    random.shuffle(pos_contrast_list)
    train_data, train_contrast_data, test_data, test_contrast_data = [], [], [], []

    for pos_data in pos_list[:pos_test_num]:
        test_data.append({'data':pos_data, 'label':1}) 
    for neg_data in neg_list[:neg_test_num]:
        test_data.append({'data':neg_data, 'label':0})

    for pos_data in pos_contrast_list[:pos_test_num]:
        test_contrast_data.append({'data':pos_data, 'label':1}) 
    for neg_data in neg_list[:neg_test_num]:
        test_contrast_data.append({'data':neg_data, 'label':0})

    for pos_data in pos_list[pos_test_num:]:
        train_data.append({'data':pos_data, 'label':1}) 
    for pos_data in pos_contrast_list[pos_test_num:]:
        train_contrast_data.append({'data':pos_data, 'label':1}) 

    for neg_data in neg_list[neg_test_num:]:
        train_data.append({'data':neg_data, 'label':0})
        train_contrast_data.append({'data':neg_data, 'label':0}) 
    return train_data, train_contrast_data, test_data, test_contrast_data





if __name__=="__main__":
    args = parse_args()
    data_root = args.data_root
    save_path = args.save_dir
    test_size = args.test_size
    same_action = bool(args.same_action)
    pos_contain_action = args.pos_contain_action
    neg_mask_action = args.neg_mask_action
    emb_dict = {}
    print(os.listdir(data_root))
    for pkl in os.listdir(data_root):
        cls_p = ' '.join(pkl.split('_')[:-2])
        print('loading {}'.format(cls_p))
        emb_dict[cls_p] = load_pkl(os.path.join(data_root, pkl))
    class_name_list = list(emb_dict.keys())

    if isinstance(pos_contain_action, int):
        pos_contain_action = [pos_contain_action]
    if isinstance(neg_mask_action, int):
        neg_mask_action = [neg_mask_action]

    for pos_n in pos_contain_action:
        for neg_n in neg_mask_action:
            save_normal_path = osp.join(save_path, '{}_{}_{}_{}_normal.json'.format(test_size, pos_n, neg_n, same_action))
            save_contrast_path = osp.join(save_path, '{}_{}_{}_{}_contrast.json'.format(test_size, pos_n, neg_n, same_action))
            result_dict = {}
            result_contrast_dict = {}
            for class_name in class_name_list:
                result_dict[class_name] = {}
                result_contrast_dict[class_name] = {}
                train_data, train_contrast_data, test_data, test_contrast_data = split_data(emb_dict, class_name, test_size=test_size, same_action=same_action, pos_contain_action=pos_n, neg_mask_action=neg_n)
                result_dict[class_name]['train'] = train_data
                result_contrast_dict[class_name]['train'] = train_contrast_data
                result_dict[class_name]['test'] = test_data
                result_contrast_dict[class_name]['test'] = test_contrast_data
            save_json(save_normal_path, result_dict)
            save_json(save_contrast_path, result_contrast_dict)
