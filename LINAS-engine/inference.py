import torch
import os
from model import get_model
from basic.util import read_dict
from basic.bigfile import BigFile
import util.tag_data_provider as data
import evaluation
import pickle
from util.text2vec import get_text_encoder
from util.vocab import clean_str
from util.vocab import Vocabulary
import numpy as np
import argparse

def process_cap(caption):
    if bow2vec is not None:
        cap_bow = bow2vec.mapping(caption)
        if cap_bow is None:
            cap_bow = torch.zeros(bow2vec.ndims)
        else:
            cap_bow = torch.Tensor(cap_bow)
    else:
        cap_bow = None

    if vocab is not None:
        tokens = clean_str(caption)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        cap_tensor = torch.Tensor(caption)
    else:
        cap_tensor = None

    return cap_tensor.long().unsqueeze(0), cap_bow.unsqueeze(0), [len(cap_tensor)], torch.ones(1,len(cap_tensor))

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='a man and a woman is talking.', type=str, help='input sentence')
    parser.add_argument('--topK', default=10, type=int, help='return top-k videos')
    parser.add_argument('--gpu', default='5', type=str, help='gpu device')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    checkpoint = torch.load('student_support_set_8/model_best.pth.tar')
    options = checkpoint['opt']
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'], 'test')
    model.Eiters = checkpoint['Eiters']
    model.val_start()
    rootpath = 'dataset/'

    if os.path.exists('video_data.pt'):
        video_data = torch.load('video_data.pt')
        video_embs = video_data['video_embs']
        video_ids = video_data['video_ids']
    else:
        img_feat_path = os.path.join(rootpath, options.collections_pathname['test'], 'FeatureData', options.visual_feature)
        visual_feats = BigFile(img_feat_path)
        video2frames = read_dict(os.path.join(rootpath, options.collections_pathname['test'], 'FeatureData', options.visual_feature, 'video2frames.txt'))
        vid_data_loader = data.get_vis_data_loader(visual_feats, options.batch_size, options.workers, video2frames, video_ids=list(video2frames.keys()))
        video_embs, video_ids = evaluation.encode_vid(model.embed_vis_distill, vid_data_loader)
        torch.save({'video_embs':video_embs,'video_ids':video_ids}, 'video_data.pt')

    bow_vocab_file = os.path.join(rootpath, options.collections_pathname['train'], 'TextData', 'vocabulary', 'bow', options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)

    rnn_vocab_file = os.path.join(rootpath, options.collections_pathname['train'], 'TextData', 'vocabulary', 'rnn', options.vocab+'.pkl')
    vocab = pickle.load(open(rnn_vocab_file, 'rb'))

    a = process_cap(opt.input)
    cap_emb = model.embed_txt_distill(a).data.cpu().numpy()
    errors = evaluation.cal_error(video_embs, cap_emb, options.measure)
    inds = np.argsort(errors[0])[:opt.topK]
    results = [video_ids[i] for i in inds]
    
    print(results)