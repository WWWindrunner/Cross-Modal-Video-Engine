import sys 
sys.path.append("..")
import os
import os.path as osp
import argparse
import torch
import random
from model_layers import get_model_layers
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.utils import Grad
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import json
from tqdm import tqdm
from collections import defaultdict
from concept_score import get_concept_score, get_concept_vector_embeddings_mean, get_concept_vector_embeddings_all, get_concept_vector_embeddings_cluster
from probing import TimeSformerHead, build_optimizer, top_k_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='get embeddings for different model on contrast clip pairs')
    parser.add_argument('--data_root', default='/data1/shufan/mmaction2/data/kinetics400')
    parser.add_argument('--grad_root', default='/data1/shufan/mmaction2/data/kinetics400')
    parser.add_argument('--model_name', default='timesformer', help='save binary classifier weights')
#    parser.add_argument('--video_per_classes', default=20, type=int, help='save binary classifier weights')
    parser.add_argument('--target_layer', default=-1, type=int, help='save binary classifier weights')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--n_clusters', default=40, type=int, nargs='+', help='cluster num for each concept')
    parser.add_argument('--person_cluster', default=100, type=int, nargs='+', help='cluster num for each concept')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--n_epoch', default=20, type=int, help='epoch')
    parser.add_argument('--optimizer', default='adam', help='save binary classifier weights')
    parser.add_argument('--scheduler', default='exp', help='save binary classifier weights')
    parser.add_argument('--save_model', action='store_true', help='save classifier weights')
    parser.add_argument('--save_dir', default='/data1/shufan/mmaction2/tools/clip_inference/cbm_model', help='save classifier weights')
    
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

class Residual_Head(nn.Module):

    def __init__(self,
                 emb_channels,
                 num_classes=1,
                 init_std=0.02):
        super().__init__()
        self.init_std = init_std
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.cls_1 = nn.Linear(self.in_channels, self.num_classes)
        self.cls_2 = nn.Linear(self.emb_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.cls_1, std=self.init_std)
        trunc_normal_init(self.cls_2, std=self.init_std)

    def forward(self, emb, concept_score):
        # [N, in_channels]
        #x = torch.cat([emb_raw, emb_mask], dim=-1)
        emb = self.cls_1(emb)
        concept_score = self.cls_2(concept_score)
        # [N, num_classes]
        return emb + concept_score

def collate_data(args):
    data_root = args.grad_root
    train_matrix_dict, test_matrix_dict = defaultdict(list), defaultdict(list)
    for pkl in os.listdir(data_root):
        pkl_path = os.path.join(data_root, pkl)
        pkl_data = load_pkl(pkl_path)
        data_num = int(pkl.split('.')[0].split('_')[-1]) - int(pkl.split('.')[0].split('_')[-2])
        if 'train' in pkl:
            for layer in pkl_data.keys():
#                if args.model_name != 'timesformer' and layer != 'label':
                if len(pkl_data[layer].size()) == 1:
                    pkl_data[layer] = pkl_data[layer].reshape(data_num, len(pkl_data[layer]) // data_num)
                train_matrix_dict[layer].append(pkl_data[layer])
        else:
            for layer in pkl_data.keys():
#                if args.model_name != 'timesformer':
                if len(pkl_data[layer].size()) == 1:
                    pkl_data[layer] = pkl_data[layer].reshape(data_num, len(pkl_data[layer]) // data_num)
                test_matrix_dict[layer].append(pkl_data[layer])
    for layer in train_matrix_dict.keys():
        train_matrix_dict[layer] = torch.cat(train_matrix_dict[layer], dim=0)
    for layer in test_matrix_dict.keys():
        test_matrix_dict[layer] = torch.cat(test_matrix_dict[layer], dim=0)

    return train_matrix_dict, test_matrix_dict
#def collate_data(args, data_type, random_vector=False):
#    device = 'cuda:0'
#    ann_path = '/data1/shufan/mmaction2/data/kinetics400/{}list.txt'.format(data_type)
#    if random_vector:
#        save_path = os.path.join(args.save_dir, '{}_{}_cbmdata_random.pkl'.format(args.model_name, data_type))
#    else:
#        save_path = os.path.join(args.save_dir, '{}_{}_cbmdata_weights.pkl'.format(args.model_name, data_type))
#    if os.path.exists(save_path):
#        return load_pkl(save_path)
#
#    output_module = get_model_layers(args.model_name)[args.target_layer]
#    idx2concept, concept_matrix = get_concept_vector_weights(args.model_dir, output_module, device, random_vector)
#    #idx2concept, concept_matrix = get_concept_vector_embeddings(args.model_dir, output_module, device, random_vector)
#
#    model = init_recognizer(args.config, args.checkpoint, device=device)
#    grad_getter = Grad(model, output_module)
#
#    ann_data = load_txt(ann_path)
#    random.shuffle(ann_data)
#    class_count = defaultdict(int)
#    filtered_ann_data = []
#    for path, label in ann_data:
#        if class_count[label] < args.video_per_classes:
#            filtered_ann_data.append([path, label])
#            class_count[label] += 1
#    ann_data = filtered_ann_data
#    data_dict = {'data':[], 'label':[], 'idx2concept':idx2concept}
#    for path, label in tqdm(ann_data):
#        video_path = os.path.join(args.data_root, path)
#        concept_score = get_concept_score(grad_getter, model, video_path, device, concept_matrix)
#        data_dict['data'].append(concept_score)
#        data_dict['label'].append(int(label))
#    data_dict['data'] = torch.stack(data_dict['data'], dim=0).detach().cpu()
#    save_pkl(save_path, data_dict)
#    return data_dict

class Cbm_Dataset(Dataset):
    def __init__(self, data, target_layer, model_name):
        self.data = []
        if model_name != 'slowfast':
            for vector, label in zip(data[target_layer], data['label']):
                self.data.append({'data': vector.cuda(), 'label': label.cuda()})
        else:
            for vector_slow, vector_fast, label in zip(data['module.backbone.slow_path.layer4.2'], data['module.backbone.fast_path.layer4.2'], data['label']):
                self.data.append({'data': torch.cat([vector_slow, vector_fast], dim=-1).cuda(), 'label': label.cuda()})
    def __getitem__(self, idx):
        label = self.data[idx]['label']
        data = self.data[idx]['data']
        return data, label
    def __len__(self):
        return len(self.data)

class Residual_Dataset(Dataset):
    def __init__(self, cbm, cbm_loader, concept_matrix):
        self.cbm = cbm
        self.data = {}
        self.cbm_loader = cbm_loader
        self.concept_matrix = concept_matrix.cuda()
        self.make_dataset()
    def make_dataset(self):
        self.cbm.eval()
        all_output, all_label, all_emb = [], [], []
        for emb, label in tqdm(self.cbm_loader):
            emb = emb.cuda()
            all_emb.append(emb)
            emb = emb.matmul(self.concept_matrix.T)
            output = model(emb)
            all_output.append(output.detach().cpu())
            all_label.append(label.detach().cpu())
        all_emb = torch.cat(all_emb, dim=0)
        all_output = torch.cat(all_output, dim=0)
        all_label = torch.cat(all_label, dim=0)
        
        self.data['concept_score'] = all_output
        self.data['label'] = all_label
        self.data['emb'] = all_emb
    def __getitem__(self, idx):
        label = self.data['label'][idx]
        concept_score = self.data['concept_score'][idx]
        emb = self.data['emb'][idx]
        return emb, concept_score, label
    def __len__(self):
        return len(self.data['emb'])

def elastic_loss(pred, label, model, alpha=0.5, beta=0.0001):
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(pred, label)
    l1_loss, l2_loss = 0, 0
    for param in model.parameters():
        l1_loss += torch.norm(param, p=1)
        l2_loss += torch.norm(param, p=2)
    e_loss = alpha*l1_loss + (1-alpha)*l2_loss
    return ce_loss + beta*e_loss



def train_model(train_dataloader, val_dataloader, concept_matrix, scheduler, optimizer, n_epoch, criterion, model, binary=True):
    best_loss, best_top1_acc, best_top5_acc, best_auc = 9999, 0, 0, 0
    concept_matrix = F.normalize(concept_matrix.cuda(), dim=-1)
    #print(concept_matrix.norm(dim=-1), concept_matrix.norm(dim=-1).size())
    for i in range(n_epoch):
        mean_loss = []
        model.train()
        for emb, label in tqdm(train_dataloader):
#            emb = emb.cuda()
            emb = emb.matmul(concept_matrix.T)
            model.zero_grad()
            optimizer.zero_grad()
            output = model(emb)
            label = label.reshape(-1, 1)
            loss = criterion(output, label.squeeze(), model)
            loss.backward()
            mean_loss.append(loss.item())
            optimizer.step()
        scheduler.step()
        #print('epoch {}, mean loss {}'.format(i, sum(mean_loss)/len(mean_loss)))
        best_loss = min(sum(mean_loss)/len(mean_loss), best_loss)
        
        #print(len(dataloader.dataset.test_data))
        model.eval()
        all_output, all_label = [], []
        for emb, label in tqdm(val_dataloader):
#            emb = emb.cuda()
            emb = emb.matmul(concept_matrix.T)
            label = label.reshape(-1, 1)
            output = model(emb)
            all_output.append(output.softmax(-1).detach().cpu())
            all_label.append(label.cpu())
        all_output = torch.cat(all_output, dim=0)
        all_label = torch.cat(all_label, dim=0)
#        all_label_onehot = np.array(F.one_hot(all_label.squeeze(), num_classes=all_output.size()[-1]))
        all_output = np.array(all_output)
        all_label = np.array(all_label).squeeze()
        print(all_output.shape)
        #mean_class_acc = mean_class_accuracy(all_output, all_label)
        #auc = roc_auc_score(all_label_onehot, all_output)
        auc = 0
        top1_acc, top5_acc = top_k_accuracy(all_output, all_label, topk=(1,5))
        print('epoch {}, loss {}, top1_acc {}, top5_acc {}, roc_auc {}'.format(i, sum(mean_loss)/len(mean_loss), top1_acc, top5_acc, auc))
        if best_top1_acc < top1_acc:
            best_top1_acc = top1_acc
            best_top5_acc = top5_acc
            best_auc = auc

        # save_dataset for residual training
    return best_loss, best_top1_acc, best_top5_acc, best_auc, model

def train_residual_model(train_dataloader, val_dataloader, concept_matrix, scheduler, optimizer, n_epoch, criterion, model, binary=True):
    best_loss, best_top1_acc, best_top5_acc, best_auc = 9999, 0, 0, 0
    concept_matrix = concept_matrix.cuda()
    for i in range(n_epoch):
        mean_loss = []
        model.train()
        for emb, concept_score, label in tqdm(train_dataloader):
            emb = emb.cuda()
            label = label.reshape(-1, 1)
            model.zero_grad()
            optimizer.zero_grad()
            output = model(emb)
            loss = criterion(output+concept_score.cuda(), label.cuda().squeeze())
            loss.backward()
            mean_loss.append(loss.item())
            optimizer.step()
        scheduler.step()
        #print('epoch {}, mean loss {}'.format(i, sum(mean_loss)/len(mean_loss)))
        best_loss = min(sum(mean_loss)/len(mean_loss), best_loss)
        
        #print(len(dataloader.dataset.test_data))
        model.eval()
        all_output, all_label = [], []
        for emb, concept_score, label in tqdm(val_dataloader):
            emb = emb.cuda()
            label = label.reshape(-1, 1)
            output = model(emb)
            all_output.append(output.detach().cpu()+concept_score)
            all_label.append(label)
        all_output = torch.cat(all_output, dim=0)
        all_label = torch.cat(all_label, dim=0)
        all_label_onehot = np.array(F.one_hot(all_label.squeeze(), num_classes=all_output.size()[-1]))
        all_output = np.array(all_output)
        all_label = np.array(all_label).squeeze()
        #print(all_label_onehot[0], all_output[0])
        #mean_class_acc = mean_class_accuracy(all_output, all_label)
        auc = roc_auc_score(all_label_onehot, all_output)
        top1_acc, top5_acc = top_k_accuracy(all_output, all_label, topk=(1,5))
        print('epoch {}, loss {}, top1_acc {}, top5_acc {}, roc_auc {}'.format(i, sum(mean_loss)/len(mean_loss), top1_acc, top5_acc, auc))
        best_top1_acc = max(best_top1_acc, top1_acc)
        best_top5_acc = max(best_top5_acc, top5_acc)
        best_auc = max(best_auc, auc)

        # save_dataset for residual training
    return best_loss, best_top1_acc, best_top5_acc, best_auc, model

if __name__=='__main__':
    args = parse_args()
    device = 'cuda:5'
    train_matrix_dict, test_matrix_dict = collate_data(args)
    target_layer = 'module.{}'.format(get_model_layers(args.model_name)[args.target_layer])
    train_dataset = Cbm_Dataset(train_matrix_dict, target_layer, args.model_name)
    val_dataset = Cbm_Dataset(test_matrix_dict, target_layer, args.model_name)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    output_module = get_model_layers(args.model_name)
    print('target layer {}'.format(output_module[args.target_layer]))

    data_root = args.data_root
    emb_dict = {}
    for pkl in os.listdir(data_root):
        cls_p = pkl.split('_')[0]
        emb_dict[cls_p] = load_pkl(os.path.join(data_root, pkl))
    if isinstance(args.n_clusters, int):
        args.n_clusters = [args.n_clusters]
    for n_clu in args.n_clusters:
        #idx2concept, concept_matrix = get_concept_vector_weights(args.model_dir, output_module, device)
        idx2concept, concept_matrix, _ = get_concept_vector_embeddings_cluster(emb_dict, output_module[args.target_layer], device, args.model_name, n_clusters=n_clu, person_cluster=n_clu)

        
        
        model = TimeSformerHead(in_channels=concept_matrix.size()[0], num_classes=400).cuda()
        scheduler, optimizer = build_optimizer(args.optimizer, args.scheduler, model.parameters(), lr=args.lr)



        best_loss, best_top1_acc, best_top5_acc, best_auc, model = train_model(train_dataloader, val_dataloader, concept_matrix, scheduler, optimizer, args.n_epoch, elastic_loss, model, binary=False)
        print('loss:{}, top1_acc:{}, top5_acc:{}, best_auc:{}'.format(best_loss, best_top1_acc, best_top5_acc, best_auc))
        if args.save_model:
#            ver = '{}_{}_{}'.format(n_clu, n_clu, args.lr)
            ver = '{}_{}_{}_{}'.format(n_clu, n_clu, args.lr, 0.0001)
            performance = {
                'n_cluster': n_clu,
#                'person_cluster': n_clu,
                'learning_rate': args.lr,
                'loss': best_loss,
                'top1_acc': best_top1_acc,
                'top5_acc': best_top5_acc,
                'auc': best_auc,
                'alpha': 0.5,
                'beta': 0.0001
            }
            performance_path = os.path.join(save_dir, '{}_performance.json'.format(ver))
            save_json(performance_path, performance)
            model_path = os.path.join(save_dir, '{}_model.pth'.format(ver))
            torch.save(model.state_dict(), model_path)
            idx2concept_path = os.path.join(save_dir, '{}_idx2concept.json'.format(ver))
            save_json(idx2concept_path, idx2concept)
            concept_matrix_path = os.path.join(save_dir, '{}_concept_matrix.pth'.format(ver))
            torch.save(concept_matrix, concept_matrix_path)
            
        #train_residual_dataset = Residual_Dataset(model, train_dataloader, concept_matrix)
        #val_residual_dataset = Residual_Dataset(model, val_dataloader, concept_matrix)
        #train_residual_dataloader = DataLoader(train_residual_dataset, batch_size=args.batch_size, shuffle=True)
        #val_residual_dataloader = DataLoader(val_residual_dataset, batch_size=args.batch_size, shuffle=True)
        #residual_loss = nn.CrossEntropyLoss()
        #residual_model = TimeSformerHead(in_channels=concept_matrix.size()[1], num_classes=400).cuda()
        #scheduler, optimizer = build_optimizer(args.optimizer, args.scheduler, residual_model.parameters(), lr=0.001)

        #best_loss, best_top1_acc, best_top5_acc, best_auc, residual_model = train_residual_model(train_residual_dataloader, val_residual_dataloader, concept_matrix, scheduler, optimizer, 50, residual_loss, residual_model, binary=False)
        #print('loss:{}, top1_acc:{}, top5_acc:{}, best_auc:{}'.format(best_loss, best_top1_acc, best_top5_acc, best_auc))
