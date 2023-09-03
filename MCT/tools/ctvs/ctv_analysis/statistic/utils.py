import os
import pickle

def process_med_feats(feats_dict, model_type='transformer', num_crops=3, level='emb'):
    avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    if model_type == "videoclip":
        num_crops = 1
    for key in feats_dict.keys():
        if isinstance(feats_dict[key], tuple):
            feats_dict[key] = feats_dict[key][0]
        feat_size = feats_dict[key].size()
        if level == 'neural':
            feats_dict[key] = feats_dict[key]
        elif len(feat_size) == 5:
            feats_dict[key] = avg_pool(feats_dict[key]).squeeze()
        elif len(feat_size) == 3:
            feats_dict[key] = feats_dict[key][:, 0]
        elif len(feat_size) == 2 and model_type == 'videoclip':
            feats_dict[key] = feats_dict[key].mean(0)
        elif len(feat_size) == 2:
            feats_dict[key] = feats_dict[key]
        else:
            print('error feat size {}'.format(feat_size))
        if level == 'emb':
            feats_dict[key] = feats_dict[key].reshape(-1, num_crops, feats_dict[key].size()[-1]).mean(1).detach().cpu()
            feats_dict[key] = feats_dict[key].mean(0)
        elif level == 'neural':
            feats_dict[key] = feats_dict[key].mean(0).detach().cpu()
        #print(feats_dict[key].size())
    return feats_dict


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
    return data

def load_video2path_dict(file_path):
    data_root = osp.dirname(file_path)
    data = load_txt(file_path)
    video2path = []
    for item in tqdm(data):
        video_name = osp.basename(item.split(' ')[0])
        video_name = video_name.split('.')[0]        
        video2path.append((video_name, osp.join(data_root, item.split(' ')[0])))
    return dict(video2path)

def save_json(save_path, data):
    with open(save_path, 'w') as f:
        json.dump(data, f)



