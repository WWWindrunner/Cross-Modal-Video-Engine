import sys 
sys.path.append("..")
import os
import random
import pickle
import numpy as np
import torch
import json
import argparse
from model_layers import get_model_layers

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f)

def parse_args():
    parser = argparse.ArgumentParser(description='get embeddings for different model on contrast clip pairs')
    parser.add_argument('--data_root', default='/data1/shufan/mmaction2/data/kinetics400/model_embeddings/entity/timesformer/val_256_mask')
    parser.add_argument('--save_path', default='/data1/shufan/mmaction2/tools/clip_inference/dataset_split.json')
    
    args = parser.parse_args()
    return args

def get_pos_list(emb_dict, class_name):
    pos_list = [video_name for video_name in emb_dict[class_name].keys()]
    return pos_list

def get_neg_list(emb_dict, class_name, pos_list):
    neg_list = []
    neg_class_list = [key for key in emb_dict.keys() if class_name != key]
    num_samples_per_cls = max(len(pos_list) // len(neg_class_list), 1)
    for neg_cls in neg_class_list[:-1]:
        neg_cls_data = emb_dict[neg_cls]
        neg_videos = ['{}->{}'.format(neg_cls, video_name) for video_name in neg_cls_data.keys()]
        neg_list += random.sample(neg_videos, min(num_samples_per_cls, len(neg_videos)))
    neg_dir = neg_class_list[-1]
    neg_cls_data = emb_dict[neg_cls]
    neg_videos = ['{}->{}'.format(neg_cls, video_name) for video_name in neg_cls_data.keys()]
    neg_list += random.sample(neg_videos, min(len(neg_videos), max(len(pos_list)-len(neg_list), 0)))
    return neg_list

def split_data(emb_dict, class_name, test_size=0.2):
    pos_list = get_pos_list(emb_dict, class_name)
    neg_list = get_neg_list(emb_dict, class_name, pos_list)
    
    pos_test_num = int(len(pos_list)*test_size)
    neg_test_num = int(len(neg_list)*test_size)
    print('{}, pos_num {}, neg_num {}'.format(class_name, len(pos_list), len(neg_list)))
    random.shuffle(pos_list)
    random.shuffle(neg_list)
    train_data, test_data = [], []
    for pos_data in pos_list[:pos_test_num]:
        test_data.append({'data':pos_data, 'label':1}) 
    for neg_data in neg_list[:neg_test_num]:
        test_data.append({'data':neg_data, 'label':0})

    for pos_data in pos_list[pos_test_num:]:
        train_data.append({'data':pos_data, 'label':1}) 
    for neg_data in neg_list[neg_test_num:]:
        train_data.append({'data':neg_data, 'label':0})
    return train_data, test_data





if __name__=="__main__":
    args = parse_args()
    data_root = args.data_root
    save_path = args.save_path
    emb_dict = {}
    for pkl in os.listdir(data_root):
        cls_p = ' '.join(pkl.split('_')[:-2])
        emb_dict[cls_p] = load_pkl(os.path.join(data_root, pkl))
    class_name_list = list(emb_dict.keys())
    result_dict = {}
    for class_name in class_name_list:
        result_dict[class_name] = {}
#        train_data, test_data = split_data(emb_dict, class_name, test_size=0.2)
        train_data, test_data = split_data(emb_dict, class_name, test_size=0)
        result_dict[class_name]['train'] = train_data
        result_dict[class_name]['test'] = test_data
    save_json(save_path, result_dict)