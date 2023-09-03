import sys 
sys.path.append("..")
import os
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import json
import argparse
from model_layers import get_model_layers
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
def save_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f)
def save_pkl(save_path, feats):
    with open(save_path, 'wb') as f:
        pickle.dump(feats, f)


def parse_args():
    parser = argparse.ArgumentParser(description='get embeddings for different model on contrast clip pairs')
    parser.add_argument('--data_root', default='/data/shufan/shufan/mmaction2/data/kinetics400/model_embeddings/entity/timesformer')
    parser.add_argument('--save_path', default='/data/shufan/shufan/mmaction2/tools/clip_inference/timesformer_entity.json')
    parser.add_argument('--model_name', default='timesformer', help='model_name')
    parser.add_argument('--task_type', default='spatial', help='spatial or temporal')
    parser.add_argument('--data_split_file', default='/data1/shufan/mmaction2/tools/clip_inference/dataset_split_alltrain_false_20.json', help='save binary classifier weights')
    parser.add_argument('--save_model_dir', default='/data1/shufan/mmaction2/data/kinetics400/concept_vector', help='save binary classifier weights')
    
    args = parser.parse_args()
    return args


class Clip_Dataset(Dataset):
    def __init__(self, layer, class_name, emb_dict, data_split, mode='train', task_type='spatial'):
        self.layer = layer
        self.class_name = class_name
        self.emb_dict = emb_dict
        self.data_split = data_split
        self.mode = mode
        self.task_type = task_type
        self.make_dataset()
    def get_inchannels(self):
        return self.data[0]['data'].size()[-1]
    def get_numclasses(self):
        return self.num_classes
    def make_dataset(self):
        self.data = []
        ann_info = self.data_split[self.class_name][self.mode]
        for item in ann_info:
            video_name = item['data']
            label = item['label']
            cls_name = ''
            if label == 1:
                cls_name = self.class_name
#                item_emb_dict = self.emb_dict[self.class_name][video_name]
            else:
                cls_name, video_name = video_name.split('->')[0], video_name.split('->')[1]
            item_emb_dict = self.emb_dict[cls_name][video_name]
            #print(list(item_emb_dict['feature'].keys()))
            if 'feature' in item_emb_dict.keys():
                if 'slow' in self.layer:
                    emb_slow = item_emb_dict['feature'][self.layer].squeeze()
                    emb_fast = item_emb_dict['feature'][self.layer.replace('slow', 'fast')].squeeze()
                    emb = torch.cat([emb_slow, emb_fast], dim=-1)
                else:
                    emb = item_emb_dict['feature'][self.layer].squeeze()
            else:
                if self.layer in item_emb_dict.keys():
                    if 'slow' in self.layer:
                        emb_slow = item_emb_dict[self.layer].squeeze()
                        emb_fast = item_emb_dict[self.layer.replace('slow', 'fast')].squeeze()
                        emb = torch.cat([emb_slow, emb_fast], dim=-1)
                    else:
                        emb = item_emb_dict[self.layer].squeeze()
                else:
                    if 'slow' in self.layer:
                        emb_slow = item_emb_dict['module.{}'.format(self.layer)].squeeze()
                        emb_fast = item_emb_dict['module.{}'.format(self.layer.replace('slow', 'fast'))].squeeze()
                        emb = torch.cat([emb_slow, emb_fast], dim=-1)
                    else:
                        emb = item_emb_dict['module.{}'.format(self.layer)].squeeze()
            emb = F.normalize(emb, dim=-1)
            self.data.append({'data': emb, 'label': label, 'concept_name': cls_name})
            self.num_classes = 1
            #if self.task_type == 'spatial':
            #    self.num_classes = 1
            #    for temporal_idx in item_emb_dict.keys():
            #        emb = item_emb_dict[temporal_idx][self.layer]
            #        self.data.append({'data': emb, 'label': label})
            #else:
            #    self.num_classes = len(item_emb_dict.keys())
            #    for temporal_idx in item_emb_dict.keys():
            #        if label == 1:
            #            for temporal_idx in item_emb_dict.keys():
            #                emb = item_emb_dict[temporal_idx][self.layer]
            #                self.data.append({'data': emb, 'label': temporal_idx})
            
    def __getitem__(self, idx):
        label = self.data[idx]['label']
        #data = self.data[idx]['data']
        data = F.normalize(self.data[idx]['data'], dim=-1)
        concept_name = self.data[idx]['concept_name']
        if self.task_type == 'spatial':
            return data, torch.Tensor([label]), concept_name
        else:
            return data, label, concept_name
    def __len__(self):
        return len(self.data)





def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


class TimeSformerHead(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes=1,
                 init_std=0.02,
                 bias=True):
        super().__init__()
        self.init_std = init_std
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes, bias=bias)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # [N, in_channels]
        #x = torch.cat([emb_raw, emb_mask], dim=-1)
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

def build_optimizer(optimizer, scheduler, params, weight_decay=0.0, lr=0.08, opt_decay_step=40, opt_decay_rate=0.99, opt_restart=1):
    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if scheduler == 'none':
        return None, optimizer
    elif scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt_decay_step, gamma=opt_decay_rate)
    elif scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_restart)
    elif scheduler == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt_decay_rate, last_epoch=-1)
    return scheduler, optimizer

def train_model(train_dataloader, test_dataloader, scheduler, optimizer, n_epoch, criterion, model):
    best_loss, best_top1_acc, best_auc = 9999, 0, 0
    best_output, best_label, best_concept_name = [], [], []
    for i in range(n_epoch):
        mean_loss = []
        model.train()
        for emb, label, _ in train_dataloader:
            model.zero_grad()
            optimizer.zero_grad()
            output = model(emb.cuda())
            loss = criterion(output, label.cuda())
            loss.backward()
            mean_loss.append(loss.item())
            optimizer.step()
        scheduler.step()
        #print('epoch {}, mean loss {}'.format(i, sum(mean_loss)/len(mean_loss)))
        best_loss = min(sum(mean_loss)/len(mean_loss), best_loss)
        
        model.eval()
        all_output, all_label, all_concept_name = [], [], []
        for emb, label, concept_name in test_dataloader:
            output = model(emb.cuda())
            all_output.append(output.detach().cpu())
            all_label.append(label)
            all_concept_name.append(concept_name)
        all_output = np.array(torch.cat(all_output, dim=0))
        all_label = np.array(torch.cat(all_label, dim=0))
        #mean_class_acc = mean_class_accuracy(all_output, all_label)
        #top1_acc = top_k_accuracy(all_output.squeeze(), all_label.squeeze())[0]
        thres = 0.5
        top1_acc = accuracy_score(all_label, all_output > thres)
        best_top1_acc = max(best_top1_acc, top1_acc)
        if all_label.shape != all_output.shape:
            all_label = np.eye(all_output.shape[-1])[all_label]
        auc = roc_auc_score(all_label, all_output)
        if best_auc < auc:
            best_auc = auc
            best_output = all_output
            best_label = all_label
            best_concept_name = all_concept_name
        best_auc = max(best_auc, auc)
        #print('epoch {}, auc {}, top1_acc {}'.format(i, auc, top1_acc))
    return best_loss, best_top1_acc, best_auc, best_output, best_label, best_concept_name, model

if __name__=="__main__":
    args = parse_args()
    data_root = args.data_root
    save_path = args.save_path
    task_type = args.task_type
    if not os.path.exists(save_path):
        emb_dict = {}
        for pkl in os.listdir(data_root):
            cls_p = ' '.join(pkl.split('_')[:-2])
            emb_dict[cls_p] = load_pkl(os.path.join(data_root, pkl))
        data_split = load_json(args.data_split_file)

        class_name_list = list(emb_dict.keys())
        layer_list = get_model_layers(args.model_name)
        batch_size = 128
        n_epoch = 10
        if task_type == 'spatial':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        result_dict = {}
        if not os.path.exists(args.save_model_dir):
            os.mkdir(args.save_model_dir)

        for class_name in class_name_list:
            #class_name = 'person'
            result_dict[class_name] = {}
            for layer in layer_list:
                train_dataset = Clip_Dataset(layer, class_name, emb_dict, data_split, mode='train', task_type=task_type)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#                test_dataset = Clip_Dataset(layer, class_name, emb_dict, data_split, mode='test', task_type=task_type)
                test_dataset = Clip_Dataset(layer, class_name, emb_dict, data_split, mode='train', task_type=task_type)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                #dataloader.dataset.change_layer(layer)
                in_channels = train_dataset.get_inchannels()
                num_classes = train_dataset.get_numclasses()
                model = TimeSformerHead(in_channels=in_channels, num_classes=num_classes).cuda()
                optimizer = 'adam'
                scheduler = 'exp'    

                scheduler, optimizer = build_optimizer(optimizer, scheduler, model.parameters(), lr=0.01)
                best_loss, best_top1_acc, best_auc, best_output, best_label, best_concept_name, model = train_model(train_dataloader, test_dataloader, scheduler, optimizer, n_epoch, criterion, model)
                
                #model_save_path = os.path.join(args.save_model_dir, "{}_{}_denoise_10.pth".format(class_name, layer))
                #torch.save(model.state_dict(), model_save_path)
                #result_dict[class_name][layer] = {'best_loss':best_loss, 'best_top1_acc':best_top1_acc, 'best_auc': best_auc, 'output': best_output, 'label': best_label, 'concept_name': best_concept_name}
                result_dict[class_name][layer] = {'best_loss':best_loss, 'best_top1_acc':best_top1_acc, 'best_auc': best_auc}
                print('class_name:{}, layer:{}, loss:{}, top1_acc:{}, best_auc:{}, in_channels:{}'.format(class_name, layer, best_loss, best_top1_acc, best_auc, in_channels))
                save_json(save_path, result_dict)
    else:
        print('{} has been probing'.format(save_path))