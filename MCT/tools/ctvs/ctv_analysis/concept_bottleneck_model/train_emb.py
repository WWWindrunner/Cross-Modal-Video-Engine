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
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import torch
import json
from tqdm import tqdm
from collections import defaultdict
from concept_score import get_concept_score, get_concept_vector_embeddings_mean, get_concept_vector_embeddings_all, get_concept_vector_embeddings_cluster
from probing import TimeSformerHead, build_optimizer, top_k_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='get embeddings for different model on contrast clip pairs')
    parser.add_argument('--grad_root', default='/data1/shufan/mmaction2/data/kinetics400')
    parser.add_argument('--model_name', default='timesformer', help='save binary classifier weights')
    parser.add_argument('--target_layer', default=-1, type=int, help='save binary classifier weights')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--n_epoch', default=20, type=int, help='epoch')
    parser.add_argument('--optimizer', default='adam', help='save binary classifier weights')
    parser.add_argument('--scheduler', default='exp', help='save binary classifier weights')
    parser.add_argument('--save_model', action='store_true', help='save classifier weights')
    parser.add_argument('--save_dir', default='/data/shufan/shufan/mmaction2/tools/clip_inference/emb_model', help='save classifier weights')
    
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


class Cbm_Dataset(Dataset):
    def __init__(self, data, target_layer, model_name):
        self.data = []
        if model_name != 'slowfast_slow':
            for vector, label in zip(data[target_layer], data['label']):
                self.data.append({'data': vector.cuda(), 'label': label.cuda()})
        else:
            for vector_slow, vector_fast, label in zip(data[target_layer], data[target_layer.replace('slow', 'fast')], data['label']):
                self.data.append({'data': torch.cat([vector_slow, vector_fast], dim=-1).cuda(), 'label': label.cuda()})
    def __getitem__(self, idx):
        label = self.data[idx]['label']
        data = self.data[idx]['data']
        return data, label
    def emb_size(self):
        return self.data[0]['data'].size()
    def __len__(self):
        return len(self.data)



def elastic_loss(pred, label, model, alpha=0, beta=0):
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(pred, label)
    l1_loss, l2_loss = 0, 0
    for param in model.parameters():
        l1_loss += torch.norm(param, p=1)
        l2_loss += torch.norm(param, p=2)
    e_loss = alpha*l1_loss + (1-alpha)*l2_loss
    return ce_loss + beta*e_loss



def train_model(train_dataloader, val_dataloader, scheduler, optimizer, n_epoch, criterion, model, binary=True):
    best_loss, best_top1_acc, best_top5_acc, best_auc = 9999, 0, 0, 0
    best_cuf_mat = None
    for i in range(n_epoch):
        mean_loss = []
        model.train()
        for emb, label in tqdm(train_dataloader):
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
            label = label.reshape(-1, 1)
            output = model(emb)
            all_output.append(output.softmax(-1).detach().cpu())
            all_label.append(label.cpu())
        all_output = torch.cat(all_output, dim=0)
        all_label = torch.cat(all_label, dim=0)
#        all_label_onehot = np.array(F.one_hot(all_label.squeeze(), num_classes=all_output.size()[-1]))
        all_output = np.array(all_output)
        all_label = np.array(all_label).squeeze()
#        print(all_output[0])
        #mean_class_acc = mean_class_accuracy(all_output, all_label)
        #auc = roc_auc_score(all_label_onehot, all_output)
        auc = 0
        top1_acc, top5_acc = top_k_accuracy(all_output, all_label, topk=(1,5))
        cuf_mat = confusion_matrix(np.argmax(all_output, axis=-1), all_label)
        print('epoch {}, loss {}, top1_acc {}, top5_acc {}, roc_auc {}'.format(i, sum(mean_loss)/len(mean_loss), top1_acc, top5_acc, auc))
        if best_top1_acc < top1_acc:
            best_top1_acc = top1_acc
            best_top5_acc = top5_acc
            best_auc = auc
            best_cuf_mat = cuf_mat

        # save_dataset for residual training
    return best_loss, best_top1_acc, best_top5_acc, best_auc, model, best_cuf_mat



if __name__=='__main__':
    args = parse_args()
    device = 'cuda:5'
    train_matrix_dict, test_matrix_dict = collate_data(args)
    output_module = get_model_layers(args.model_name)
    #print('target layer {}'.format(output_module[args.target_layer]))
    for target_layer in output_module:
        print('target layer {}'.format(target_layer))

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        save_dir = os.path.join(args.save_dir, args.model_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        ver = '{}_{}_{}'.format(target_layer, args.lr, 0.0001)
        performance_path = os.path.join(save_dir, '{}_performance.json'.format(ver))
        mat_path = os.path.join(save_dir, '{}_mat.pkl'.format(ver))
        model_path = os.path.join(save_dir, '{}_model.pth'.format(ver))

        if os.path.exists(performance_path):
            print('{} exists'.format(performance_path))
#            continue

        train_dataset = Cbm_Dataset(train_matrix_dict, 'module.{}'.format(target_layer), args.model_name)
        val_dataset = Cbm_Dataset(test_matrix_dict, 'module.{}'.format(target_layer), args.model_name)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        
        model = TimeSformerHead(in_channels=train_dataset.emb_size()[0], num_classes=400, bias=False).cuda()
        scheduler, optimizer = build_optimizer(args.optimizer, args.scheduler, model.parameters(), lr=args.lr)

        best_loss, best_top1_acc, best_top5_acc, best_auc, model, best_cuf_mat = train_model(train_dataloader, val_dataloader, scheduler, optimizer, args.n_epoch, elastic_loss, model, binary=False)
        print('loss:{}, top1_acc:{}, top5_acc:{}, best_auc:{}'.format(best_loss, best_top1_acc, best_top5_acc, best_auc))
        if args.save_model:
            
            performance = {
                'learning_rate': args.lr,
                'loss': best_loss,
                'top1_acc': best_top1_acc,
                'top5_acc': best_top5_acc,
                'auc': best_auc,
                'alpha': 0,
                'beta': 0
            }
            save_json(performance_path, performance)
            save_pkl(mat_path, best_cuf_mat)
            torch.save(model.state_dict(), model_path)
        
