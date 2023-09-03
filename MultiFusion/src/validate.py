import multiprocessing
import re
import time
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

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
from utils import extract_index_features, collate_fn, element_wise_sum, device



def compute_cirr_val_metrics(relative_val_dataset: ComposedVideoDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable, combiner) -> Tuple[
    float, float, float, float, float, float, float]:
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
    predicted_features, reference_names, target_names = \
        generate_cirr_val_predictions(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    with torch.no_grad():
        b = 128
        index_features_tmp = []
        for bt_index in range(int(len(index_features) / b) + 1):
            if bt_index < int(len(index_features) / b):
                index_features_tmp.append(combiner.time_process(index_features[bt_index * b:(bt_index + 1) * b, ]))

            else:
                index_features_tmp.append(combiner.time_process(index_features[bt_index * b:, ]))
        index_features = torch.cat(index_features_tmp, dim=0)

    index_features = F.normalize(index_features, dim=-1).float()
    predicted_features = predicted_features
    print("Compute the distances and sort the results")

    print("Compute CIRR validation metrics")

    # Normalize the index features
    # combiner.eval()
    # with torch.no_grad():
    #     index_features = combiner.time_process(index_features)
    b = 32
    #     distances_tmp = []
    sorted_indices_tmp = []
    reference_mask = []
    sorted_index_names = []
    labels = []
    for bt_index in tqdm(range(int(len(predicted_features) / b) + 1)):
        if bt_index < int(len(predicted_features) / b):
            tmp = 1 - predicted_features[bt_index * b:(bt_index + 1) * b, ] @ index_features.T
            sorted_indices_tmp = torch.argsort(tmp.cpu(), dim=-1)
            sorted_index_names_tmp = torch.tensor(index_names)[sorted_indices_tmp]
            reference_mask_tmp = torch.tensor(
                sorted_index_names_tmp != torch.tensor(reference_names[bt_index * b:(bt_index + 1) * b]).unsqueeze(
                    dim=1).repeat(1, len(index_names)).reshape(len(target_names[bt_index * b:(bt_index + 1) * b]), -1))
            #             reference_mask_tmp = torch.ones_like(reference_mask_tmp).bool()
            #                 index_features_tmp.append(combiner.time_process(index_features[bt_index*b:(bt_index+1)*b,]))
            sorted_index_names_tmp = sorted_index_names_tmp[reference_mask_tmp].reshape(sorted_index_names_tmp.shape[0],
                                                                                        sorted_index_names_tmp.shape[
                                                                                            1] - 1)
            labels_tmp = torch.tensor(
                sorted_index_names_tmp[:, :50] == torch.tensor(target_names[bt_index * b:(bt_index + 1) * b]).unsqueeze(
                    dim=1).repeat(1, 50).reshape(
                    len(target_names[bt_index * b:(bt_index + 1) * b]), -1))

        else:
            tmp = 1 - predicted_features[bt_index * b:, ] @ index_features.T
            #         distances_tmp.append(tmp.cpu())
            sorted_indices_tmp = torch.argsort(tmp.cpu(), dim=-1)
            sorted_index_names_tmp = torch.tensor(index_names)[sorted_indices_tmp]
            reference_mask_tmp = torch.tensor(
                sorted_index_names_tmp != torch.tensor(reference_names[bt_index * b:]).unsqueeze(dim=1).repeat(1,
                                                                                                               len(index_names)).reshape(
                    len(target_names[bt_index * b:]), -1))
            #             reference_mask_tmp = torch.ones_like(reference_mask_tmp).bool()
            sorted_index_names_tmp = sorted_index_names_tmp[reference_mask_tmp].reshape(sorted_index_names_tmp.shape[0],
                                                                                        sorted_index_names_tmp.shape[
                                                                                            1] - 1)
            labels_tmp = torch.tensor(
                sorted_index_names_tmp[:, :50] == torch.tensor(target_names[bt_index * b:]).unsqueeze(dim=1).repeat(1,
                                                                                                                    50).reshape(
                    len(target_names[bt_index * b:]), -1))
        #         reference_mask.append(reference_mask_tmp.cpu())
        #         sorted_index_names.append(sorted_index_names_tmp.cpu())
        labels.append(labels_tmp)
        sorted_index_names.append(sorted_index_names_tmp)
    #     reference_mask = torch.cat(reference_mask,dim=0)
    #     sorted_index_names = torch.cat(sorted_index_names,dim=0)
    labels = torch.cat(labels, dim=0)
    sorted_index_names = torch.cat(sorted_index_names, dim=0)

    #     sorted_indices = torch.cat(sorted_indices_tmp,dim=0)
    #                 index_features_tmp.append(combiner.time_process(index_features[bt_index*b:,]))
    #     sorted_indices = torch.argsort(distances, dim=-1).cpu()
    #     print("sorted_indices success")
    np.save("results_wo_attn", sorted_index_names[:,:100])
    # Delete the reference image from the results
    #     reference_mask = torch.tensor(
    #         sorted_index_names != torch.tensor(reference_names).unsqueeze(dim=1).repeat(1, len(index_names)).reshape(
    #             len(target_names), -1))
    print("reference_mask success")

    # Compute the subset predictions and ground-truth labels
    # group_members = np.array(group_members)
    # group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    # group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    # assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    # assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = -1
    group_recall_at2 = -1
    group_recall_at3 = -1

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50

def get_strs(type):
    same_obj_diff_attr = []
    train_pt = "/data/yuewu/data/action_genome_v1.0/modified_dataset/caption_by_CLIP__objects_scenes_attributes_threshold_0.17_version_5/vdo_modified_text_{}_clip_remaped.txt".format(type)
    train_data = open(train_pt, 'r')
    train_data_all = train_data.readlines()


    for info in train_data_all:
        cap = info.split('\t')[-1]
        # if 'the scene where the ' in cap:
        #     m = list(re.match("the scene where the (.+) is in changes to the (.+)", cap).groups())
        #     for v in m:
        #         same_obj_diff_scene.append(v.lower())
        # elif 'but with' in cap:
        #     m = list(re.match("the (.+) is same but with (.+)", cap).groups())
        #     for v in m:
        #         same_scene_diff_obj.append(v.lower())
        if 'the attribute of the 'in cap:
            m = list(re.match("the attribute of the (.+) is replaced by (.+)", cap).groups())
            # for v in m:
            same_obj_diff_attr.append(m[-1].lower())
    return same_obj_diff_attr
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
    # dataset_root = "/data/yuewu/data/action_genome_v1.0/"
    # ag_anno_train = pd.read_csv(os.path.join(dataset_root, "Charades", "Charades_v1_train.csv"), index_col='id')
    # ag_anno_test = pd.read_csv(os.path.join(dataset_root, "Charades", "Charades_v1_test.csv"), index_col='id')
    # ag_anno = pd.concat([ag_anno_train, ag_anno_test])
    # print("original data len is {}".format(len(ag_anno)))
    # for k in ag_anno.columns:
    #     ag_anno = ag_anno[ag_anno[k].notna()]
    # ag_acts = np.load(os.path.join(dataset_root, 'clipped_vdo_action.npy'), allow_pickle=True).item()
    # ag_acts_df = pd.DataFrame.from_dict(ag_acts, orient='index')
    # ag_acts_cls = ag_acts_df[0].unique().tolist()
    # print("remove the nan objects and scene..\n remained {}".format(len(ag_anno)))
    # obj_cls = pd.read_table(os.path.join(dataset_root, "Charades",  "Charades_v1_objectclasses.txt"), header=None, sep=' ')[1:][1].values.tolist()
    # event_cls = pd.read_table(os.path.join(dataset_root, "Charades",  "Charades_v1_classes_table.txt"), header=None, sep='\t')[1].values.tolist()
    # all_cls = obj_cls.copy()
    # scene_cls = ag_anno['scene'].str.lower().unique().tolist()[:-1]
    # attribute_cls = set(get_strs('test'))
    # all_cls.extend(scene_cls)
    # all_cls.extend(attribute_cls)
    # all_cls.extend(event_cls)
    # all_cls = list(set(all_cls))
    # with torch.no_grad():
    #     text = clip.tokenize(all_cls).to(device)
    #     all_text_features = clip_model.encode_text(text)
    #     all_text_features /= all_text_features.norm(dim=-1, keepdim=True)

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=4,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []
    #
    # obj_cnt = 0
    # scene_cnt = 0
    # attribute_cnt = 0
    # # event_cnt = 0
    # objs = []
    # scenes = []
    # attributes = []
    # # events = []
    for batch_reference_names, batch_target_names, captions, batch_group_members, middle_feature in tqdm(
            relative_val_loader):  # Load data
        text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        # batch_group_members = np.array(batch_group_members).T.tolist()
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
            # batch_predicted_features = combining_function(reference_image_features.mean(dim=1), text_features)

            # after_temp_ref_high_feature /= after_temp_ref_high_feature.norm(dim=-1, keepdim=True)
            # similarity = (100.0 * batch_predicted_features  @ all_text_features.T).softmax(dim=-1)
            # for i in range(similarity.shape[0]):
            #     values, indices = similarity[i].topk(1)
            #     text_value = all_cls[indices]
            #     if text_value in scene_cls:
            #         scene_cnt += 1
            #         scenes.append(text_value)
            #     if text_value in attribute_cls:
            #         attribute_cnt += 1
            #         attributes.append(text_value)
            #     if text_value in obj_cls:
            #         obj_cnt += 1
            #         objs.append(text_value)
                # if text_value in event_cls:
                #     event_cnt += 1
                #     events.append(text_value)
        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        # group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
    # print("obj_cnt", obj_cnt / (obj_cnt+scene_cnt+attribute_cnt))
    # print("scene_cnt", scene_cnt / (obj_cnt+scene_cnt+attribute_cnt))
    # print("attribute_cnt", attribute_cnt / (obj_cnt+scene_cnt+attribute_cnt))
    # print("event_cnt", event_cnt / (obj_cnt+scene_cnt+attribute_cnt+event_cnt))
    # np.save("mean_after_td.py", {'objs':objs, 'attributes':attributes, 'scenes':scenes})
    # np.save("lstm_rst.py", {'objs':objs, 'attributes':attributes, 'scenes':scenes})
    return predicted_features, reference_names, target_names


def cirr_val_retrieval(combining_function: callable, clip_model: CLIP, preprocess: callable, args, combiner):
    """
    Perform retrieval on CIRR validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """

    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = ComposedVideoDataset('test', 'classic', preprocess, args.data_pth, args.dataset_op)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = ComposedVideoDataset('test', 'relative', preprocess, args.data_pth, args.dataset_op)

    return compute_cirr_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                    combining_function, combiner)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='ComposedVideo', type=str, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--data_pth", type=str, default="/data/yuewu/data/composed_dataset",
                        help="the dataset's path")
    parser.add_argument("--dataset_op", type=str,
                        default="caption_by_CLIP__objects_scenes_attributes_threshold_0.17_version_10",
                        help="the dataset's option")


    parser.add_argument("--combining-function", type=str, default='combiner',
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path,  default="/data/yuewu/projects/video_retrieval/Version2/CLIP4Cvr_multilayers_composed_data/models/version_10/RN50x4/cvpr_22_middle_trans_fc_kc_qt_vc_relu_dropout_lxf_output_fusion_finnal_wo_attn_no_cnn_no_concat_only_plus/saved_models/combiner_arithmetic.pt", help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path",type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")

    args = parser.parse_args()

    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
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

    if args.dataset.lower() == 'composedvideo':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(combining_function, clip_model, preprocess, args, combiner)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    elif args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', combining_function, clip_model,
                                                                     preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', combining_function, clip_model,
                                                                     preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', combining_function, clip_model,
                                                                       preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("time cost", time.time()-t0)