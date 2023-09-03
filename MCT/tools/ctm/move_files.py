import os
import shutil
def get_class_video_dict(train_root):
    result_dict = dict()
    for class_name in os.listdir(train_root):
        result_dict[class_name] = set(os.listdir(os.path.join(train_root, class_name)))
    return result_dict

def remove_files_by_classname(class_video_dict, mask_root, target_root):
    for class_name, video_set in class_video_dict.items():
        class_root = os.path.join(target_root, class_name)
        if not os.path.exists(class_root):
            os.mkdir(class_root)
        for concept_name in os.listdir(mask_root):
            class_concept_root = os.path.join(class_root, concept_name)
            if not os.path.exists(class_concept_root):
                os.mkdir(class_concept_root)
    for concept_name in os.listdir(mask_root):
        concept_dir = os.path.join(mask_root, concept_name)
        for video_file in os.listdir(concept_dir):
            video_path = os.path.join(concept_dir, video_file)
            for class_name, video_set in class_video_dict.items():
                if video_file in video_set:
                    new_video_path = os.path.join(target_root, class_name, concept_name, video_file)
                    print('move {} to {}'.format(video_path, new_video_path))
                    shutil.copyfile(video_path, new_video_path)
                    break

if __name__=="__main__":
    train_root = "/data/shufan/shufan/mmaction2/data/kinetics400/train_256"
    mask_root = "/data/shufan/shufan/mmaction2/data/kinetics400/train_256_yolov5s_mask"
    target_root = "/data/shufan/shufan/mmaction2/data/kinetics400/train_256_mask"
    if not os.path.exists(target_root):
        os.mkdir(target_root)
    class_video_dict = get_class_video_dict(train_root)
    remove_files_by_classname(class_video_dict, mask_root, target_root)