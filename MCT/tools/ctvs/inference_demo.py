import argparse
import sys 
sys.path.append("..")
import os
import os.path as osp
import shutil
import json
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

from ctv_construction.model_layers import get_model_layers
from ctv_analysis.statistic.concept_score import get_concept_score

from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.utils import Grad
from mmcv.parallel import collate, scatter

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f)

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(save_path, feats):
    with open(save_path, 'wb') as f:
        pickle.dump(feats, f)
        
def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    data = [line.strip().split(' ') for line in data]
    return data

def save_txt(save_path, data):
    with open(save_path, 'w') as f:
        f.writelines(data)

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

def get_mask_video_path(data_root):
    concept_list = os.listdir(data_root)
    concept_dir_list = [os.path.join(data_root, concept) for concept in concept_list]
    video_paths = []
    for concept_dir in concept_dir_list:
        video_list = os.listdir(concept_dir)
        #concept_name = os.path.basename(concept_dir).replace(' ', '_')
        video_path_list = [os.path.join(concept_dir, video) for video in video_list]
        video_paths += video_path_list
    return video_paths

def process_med_feats(feats_dict, num_crops=3):
    avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    for key in feats_dict.keys():
        if isinstance(feats_dict[key], tuple):
            feats_dict[key] = feats_dict[key][0]
        feat_size = feats_dict[key].size()

        if len(feat_size) == 5:
            feats_dict[key] = avg_pool(feats_dict[key]).squeeze()
        elif len(feat_size) == 3:
            feats_dict[key] = feats_dict[key][:, 0]
        elif len(feat_size) == 2:
            feats_dict[key] = feats_dict[key]
        else:
            print('error feat size {}'.format(feat_size))
        feats_dict[key] = feats_dict[key].reshape(-1, num_crops, feats_dict[key].size()[-1]).mean(1).detach().cpu()
        feats_dict[key] = feats_dict[key].mean(0)
    return feats_dict

def get_ctvs(concept_matrix_root, model_name, ctv_type='instance'):
    idx2concept = load_json(osp.join(concept_matrix_root, '{}_{}_idx2concept.json'.format(model_name, ctv_type)))
    concept_matrix = torch.load(osp.join(concept_matrix_root, '{}_{}_concept_matrix.pth'.format(model_name, ctv_type)))
    if ctv_type == 'cluster':
        concept_video_dict = load_json(osp.join(concept_matrix_root, '{}_{}_concept_video_dict.json'.format(model_name, ctv_type)))
        return idx2concept, concept_matrix, concept_video_dict
    return idx2concept, concept_matrix

def get_video_ctvs(args, video_path, model_layers, target_layer):
    temp_root = args.save_dir
    video_temp_dir = osp.join(temp_root, 'video_temp')
    mask_video_temp_dir = osp.join(temp_root, 'mask_video_temp')
    if osp.exists(temp_root):
        shutil.rmtree(temp_root)
    if osp.exists(video_temp_dir):
        shutil.rmtree(video_temp_dir)
    if osp.exists(mask_video_temp_dir):
        shutil.rmtree(mask_video_temp_dir)
    os.mkdir(temp_root)
    os.mkdir(video_temp_dir)
    os.mkdir(mask_video_temp_dir)
    video_temp_path = osp.join(video_temp_dir, osp.basename(video_path))
    shutil.copyfile(video_path, video_temp_path)

    idx2concept = []
    concept_matrix = []

#    os.system('cd /data1/shufan/yolov5')
    os.system('python /data1/knowledge_graph/ssf_workspace/MCT/yolov5/detect.py --weights /data1/knowledge_graph/ssf_workspace/MCT/yolov5/weights/yolov5s.pt --source {} --project {} --exist-ok --half'.format(video_temp_dir, mask_video_temp_dir))
    os.system('python /data1/knowledge_graph/ssf_workspace/MCT/yolov5/detect_raw.py --weights /data1/knowledge_graph/ssf_workspace/MCT/yolov5/weights/yolov5s.pt --source {} --project {} --exist-ok --half'.format(video_temp_path, video_temp_dir))
    _, raw_feats = inference_recognizer(model, video_path, outputs=model_layers, return_scores=True, centercrop=False)
    raw_feats = process_med_feats(raw_feats)
    mask_video_paths = get_mask_video_path(mask_video_temp_dir)
    for mask_video_path in mask_video_paths:
        concept_name = osp.basename(osp.dirname(mask_video_path))
        _, mask_feats = inference_recognizer(model, mask_video_path, outputs=model_layers, return_scores=True, centercrop=False)
        mask_feats = process_med_feats(mask_feats)
        idx2concept.append(concept_name)
        concept_matrix.append(raw_feats[target_layer] - mask_feats[target_layer])
    concept_matrix = torch.stack(concept_matrix, dim=0).squeeze()

    return idx2concept, concept_matrix

def get_similar_concepts(args, ctv, idx2concept, concept_matrix, video2path, coco_concept2idx, ctv_type='instance', vis_num=3, concept_video_dict=None):
    concept_score = get_concept_score(ctv, concept_matrix)
    if ctv_type == 'instance':
        values, inds = concept_score.topk(vis_num, dim=0, largest=True, sorted=True)
        instance_temp_dir = osp.join(args.save_dir, 'instance_concepts_video')
        if not osp.exists(instance_temp_dir):
            os.mkdir(instance_temp_dir)
        for v, i in zip(values, inds):
            concept_name = idx2concept[str(int(i))]
            video_name = concept_name.split('->')[-1].split('.')[0]
            sym_concept_name = concept_name.split('->')[0]
            coco_idx = coco_concept2idx[sym_concept_name]
            video_path = video2path[video_name]
            os.system('python /data1/knowledge_graph/ssf_workspace/MCT/yolov5/detect_raw.py --weights /data1/knowledge_graph/ssf_workspace/MCT/yolov5/weights/yolov5s.pt --source {} --project {} --name {}_{:.2} --classes {} --exist-ok --half'.format(video_path, instance_temp_dir, sym_concept_name.replace(' ',''), v, coco_idx))
    elif ctv_type == 'symbolic':
        values, inds = concept_score.topk(concept_score.size()[0], dim=0, largest=True, sorted=True)
        symbolic_temp_file = osp.join(args.save_dir, 'symbolic_concepts_video.txt')
        symbolic_similar_data = []
        for v, i in zip(values, inds):
            symbolic_similar_data.append('{} {:.2f}\n'.format(idx2concept[str(int(i))], v))
        save_txt(symbolic_temp_file, symbolic_similar_data)
    elif ctv_type == 'cluster':
        values, inds = concept_score.topk(vis_num, dim=0, largest=True, sorted=True)
        cluster_temp_dir = osp.join(args.save_dir, 'cluster_concepts_video')
        if not osp.exists(cluster_temp_dir):
            os.mkdir(cluster_temp_dir)
        for v, i in zip(values, inds):
            concept_name = idx2concept[int(i)]
            sym_concept_name = concept_name.split('->')[0]
            video_list = concept_video_dict[concept_name][:vis_num]
            for video in video_list:
                video_name = video.split('.')[0]
                coco_idx = coco_concept2idx[sym_concept_name]
                video_path = video2path[video_name]
                os.system('python /data1/knowledge_graph/ssf_workspace/MCT/yolov5/detect_raw.py --weights /data1/knowledge_graph/ssf_workspace/MCT/yolov5/weights/yolov5s.pt --source {} --project {} --name {}_{:.2} --classes {} --exist-ok --half'.format(video_path, cluster_temp_dir, concept_name.replace('->', '_').replace(' ',''), v, coco_idx))
    else:
        raise Exception('Wrong ctv type')



def get_concept_score(grad, concept_matrix):
    concept_score = concept_matrix.matmul(grad)
    return concept_score

def build_inputs(model, video_path, use_frames=False):
    """build inputs for GradCAM.

    Note that, building inputs for GradCAM is exactly the same as building
    inputs for Recognizer test stage. Codes from `inference_recognizer`.

    Args:
        model (nn.Module): Recognizer model.
        video_path (str): video file/url or rawframes directory.
        use_frames (bool): whether to use rawframes as input.
    Returns:
        dict: Both GradCAM inputs and Recognizer test stage inputs,
            including two keys, ``imgs`` and ``label``.
    """
    if not (osp.exists(video_path) or video_path.startswith('http')):
        raise RuntimeError(f"'{video_path}' is missing")

    if osp.isfile(video_path) and use_frames:
        raise RuntimeError(
            f"'{video_path}' is a video file, not a rawframe directory")
    if osp.isdir(video_path) and not use_frames:
        raise RuntimeError(
            f"'{video_path}' is a rawframe directory, not a video file")

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    # prepare data
    if use_frames:
        filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
        modality = cfg.data.test.get('modality', 'RGB')
        start_index = cfg.data.test.get('start_index', 1)
        data = dict(
            frame_dir=video_path,
            total_frames=len(os.listdir(video_path)),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
    else:
        start_index = cfg.data.test.get('start_index', 0)
        data = dict(
            filename=video_path,
            label=-1,
            start_index=start_index,
            modality='RGB')
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    return data



def parse_args():
    parser = argparse.ArgumentParser(description='get embeddings for different model on contrast clip pairs')
    parser.add_argument('--concept_matrix_root', default='/data1/knowledge_graph/ssf_workspace/MCT/data/ctv_matrix_root/timesformer') 
    parser.add_argument('--root', default='/data1/knowledge_graph/ssf_workspace/MCT') 
    parser.add_argument('--target_video_path', default='') 
    parser.add_argument('--save_dir', default='./temp')
    parser.add_argument('--target_layer_idx', default=-1, type=int)
    parser.add_argument('--model_name', default='timesformer', help='name of action recognition models')
    
    args = parser.parse_args()
    return args

def get_config_path(model_name):
    if model_name == 'timesformer':
        return '/data1/knowledge_graph/ssf_workspace/MCT/configs/recognition/timesformer/timesformer_divST_8x32x1_15e_video_kinetics400_rgb.py'
    else:
        raise Exception("Wrong model name")

def get_checkpoint_path(model_name):
    if model_name == 'timesformer':
        return '/data1/knowledge_graph/ssf_workspace/MCT/mmaction_checkpoints/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth'
    else:
        raise Exception("Wrong model name")

def get_concept_interpretations(args, grad, video_path, model_layers, target_layer):
    idx2concept_video, concept_matrix_video = get_video_ctvs(args, video_path, model_layers, target_layer)
    concept_score = get_concept_score(grad.detach().cpu(), concept_matrix_video)
    return concept_score, idx2concept_video, concept_matrix_video

if __name__ == "__main__":
    args = parse_args()
    model_layers = get_model_layers(args.model_name)
    target_layer = model_layers[args.target_layer_idx]

    idx2class = get_idx2class('/data1/knowledge_graph/ssf_workspace/MCT/data/vallist.txt')
    class2idx = get_class2idx('/data1/knowledge_graph/ssf_workspace/MCT/data/vallist.txt')
    video2path = load_video2path_dict('/data1/knowledge_graph/ssf_workspace/MCT/data/vallist.txt')
    coco_concept2idx = load_yaml('/data1/knowledge_graph/ssf_workspace/MCT/yolov5/data/coco.yaml')['names']
    coco_concept2idx = dict([[concept, idx] for idx, concept in coco_concept2idx.items()])

    print('get multi-level CTVs...')
    idx2concept_symbolic, concept_matrix_symbolic = get_ctvs(args.concept_matrix_root, args.model_name, ctv_type='symbolic')
    idx2concept_cluster, concept_matrix_cluster, concept_video_dict_cluster = get_ctvs(args.concept_matrix_root, args.model_name, ctv_type='cluster')
    idx2concept_instance, concept_matrix_instance = get_ctvs(args.concept_matrix_root, args.model_name, ctv_type='instance')

    print('build inputs...')
    config = get_config_path(args.model_name)
    checkpoint = get_checkpoint_path(args.model_name)
    model = init_recognizer(config, checkpoint)
    inputs = build_inputs(model, args.target_video_path, use_frames=False)

    print('model inference...')
    grad_getter = Grad(model, model.cfg, model_layers)
    grad, acti, label, pred_cls = grad_getter(inputs)
    cls_str = idx2class[str(int(pred_cls))]
    grad = F.normalize(grad[target_layer][:, 0].mean(0), dim=0)

    print('generating concept-based interpretations...')
    concept_score, idx2concept_video, concept_matrix_video = get_concept_interpretations(args, grad, args.target_video_path, model_layers, target_layer)
    concept_score = [float(s) for s in concept_score]
    results_video = dict(zip(idx2concept_video, concept_score))
    save_json(osp.join(args.save_dir, 'results_video.json'), results_video)
    print('search similar concepts...')
    max_idx = concept_score.index(max(concept_score))
    ctv = concept_matrix_video[max_idx]
    get_similar_concepts(args, ctv, idx2concept_symbolic, concept_matrix_symbolic, video2path, coco_concept2idx, ctv_type='symbolic', vis_num=3, concept_video_dict=None)
    get_similar_concepts(args, ctv, idx2concept_cluster, concept_matrix_cluster, video2path, coco_concept2idx, ctv_type='cluster', vis_num=3, concept_video_dict=concept_video_dict_cluster)
    get_similar_concepts(args, ctv, idx2concept_instance, concept_matrix_instance, video2path, coco_concept2idx, ctv_type='instance', vis_num=3, concept_video_dict=None)