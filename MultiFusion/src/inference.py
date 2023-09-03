import multiprocessing
import re
import time
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple
from PIL import Image
import clip
import numpy as np
import os

import pandas as pd
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import squarepad_transform, ComposedVideoDataset, targetpad_transform
from combiner import Combiner
from utils import extract_index_features, collate_fn, element_wise_sum, device, extract_vdo_features
from model.clip import _transform, load

def compute_cirr_val_metrics(ref_vdo_feature, mod_text, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable, combiner):
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions

    # with torch.no_grad():
    #     b = 128
    #     index_features_tmp = []
    #     for bt_index in range(int(len(index_features) / b) + 1):
    #         if bt_index < int(len(index_features) / b):
    #             index_features_tmp.append(combiner.time_process(index_features[bt_index * b:(bt_index + 1) * b, ]))
    #
    #         else:
    #             index_features_tmp.append(combiner.time_process(index_features[bt_index * b:, ]))
    #     index_features = torch.cat(index_features_tmp, dim=0)

    index_features = F.normalize(index_features, dim=-1).float()
    print("Compute the distances and sort the results")

    ref_vdo_feature_high, ref_vdo_feature_middle = ref_vdo_feature
    ref_vdo_feature_high = ref_vdo_feature_high.unsqueeze(0)
    text_inputs = clip.tokenize(mod_text).to(device, non_blocking=True)
    middle_feature = ref_vdo_feature_middle.to(device, non_blocking=True).float()
    middle_feature = torch.nn.functional.adaptive_avg_pool2d(
        middle_feature.reshape(1, middle_feature.shape[0], 18 * 18, -1), (16, index_features.shape[-1]))
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        batch_predicted_features = combining_function((ref_vdo_feature_high, middle_feature), text_features)
    scores = 1- batch_predicted_features @ index_features.T
    scores_index = torch.argsort(scores.cpu(), dim=-1)
    scores_name_top_1 = index_names[scores_index[0][0]]
    return scores_name_top_1


def generate_cirr_val_predictions(clip_model: CLIP, relative_val_dataset: ComposedVideoDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print("Compute CIRR validation predictions")
    clip_model.eval()

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=4,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members, middle_feature in tqdm(
            relative_val_loader):  # Load data
        text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        middle_feature = middle_feature.to(device, non_blocking=True).float()
        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names.numpy())(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function((reference_image_features, middle_feature), text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        reference_names.extend(batch_reference_names)
    return predicted_features, reference_names, target_names


def cirr_val_retrieval(combining_function: callable, clip_model: CLIP, preprocess: callable, args, combiner, ref_vdo, mod_text, tar_list, clip_preprocess):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()
    ref_vdo_feature_high, ref_vdo_feature_middle = extract_vdo_features(ref_vdo, clip_model, clip_preprocess)
    # Define the validation datasets and extract the index features
    classic_val_dataset = ComposedVideoDataset('test', 'classic', preprocess, args.data_pth, args.dataset_op)
    index_features = torch.concat([combiner.time_process(extract_vdo_features(tar_vdo, clip_model, clip_preprocess)[0].unsqueeze(0)) for tar_vdo in tar_list])
    top1_name = compute_cirr_val_metrics((ref_vdo_feature_high, ref_vdo_feature_middle), mod_text, clip_model, index_features,
                             tar_list,
                             combining_function, combiner)
    output_pth = "../outputs"
    if not os.path.exists(output_pth):
        os.mkdir(output_pth)
    else:
        for i in os.listdir(output_pth):
            # print(i)
            txt_path = os.path.join(output_pth, i)
            os.remove(txt_path)
        os.rmdir(output_pth)
        os.mkdir(output_pth)
    cmd = "cp {} {}".format(top1_name, output_pth)
    os.system(cmd)
    print("top-1 retrieved video is saved in the {} path".format(output_pth))
    return top1_name


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='ComposedVideo', type=str, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--data_pth", type=str, default="../dataset",
                        help="the dataset's path")
    parser.add_argument("--dataset_op", type=str,
                        default="./",
                        help="the dataset's option")


    parser.add_argument("--combining-function", type=str, default='combiner',
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path,  default="checkpoint/combiner_arithmetic.pt", help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path",type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--input_vdo", default="../dataset/videos/XZM6Y.mp4", type=str,
                        help="input reference video pth")
    parser.add_argument("--input_modified_text", default="the person changes from standing and holding a sandwich to walking and holding a broom while examining a box from a shelf", type=str,
                        help="modified text")
    args = parser.parse_args()
    print("gt", "5OMSL")
    clip_model, clip_preprocess, clip_preprocess = load(args.clip_model_name, jit=False)
    clip_model = clip_model.to(device)
    if args.clip_model_path:
        print('Trying to load the fine-tuned CLIP model')
        clip_model_path = args.clip_model_path
        state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(state_dict["CLIP"])
        print('CLIP model loaded successfully')

    # clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = element_wise_sum

    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")
    tar_list = os.listdir(os.path.join(args.data_pth, 'videos'))
    tar_list = [os.path.join(args.data_pth, 'videos', tar) for tar in tar_list]
    if args.dataset.lower() == 'composedvideo':
        top_1_name = \
            cirr_val_retrieval(combining_function, clip_model, preprocess, args, combiner, args.input_vdo, args.input_modified_text, tar_list, clip_preprocess)
        print(top_1_name)
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("time cost", time.time()-t0)