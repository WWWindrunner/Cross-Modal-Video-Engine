import os

def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data
def save_txt(save_path, data):
    with open(save_path, 'w') as f:
        f.writelines(data)

if __name__=="__main__":
    train_root = "/data/shufan/shufan/mmaction2/data/kinetics400/train_256"
    mask_root = '/data/shufan/shufan/mmaction2/data/kinetics400/train_256_mask'
    # generate trainlist txt
    action_list = os.listdir(train_root)
    action_dir_list = [os.path.join(train_root, action) for action in action_list]
    save_dir = '/data/shufan/shufan/mmaction2/data/kinetics400/trainlist_classwise_for_inference'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for action_dir in action_dir_list:
        video_list = os.listdir(action_dir)
        video_path_list = [os.path.join(action_dir, video) for video in video_list]
        write_data = ['{} 0\n'.format(video_path) for video_path in video_path_list]
        save_path = os.path.join(save_dir, '{}.txt'.format(os.path.basename(action_dir)))
        save_txt(save_path, write_data)

    # generate concept list txt
    action_list = os.listdir(mask_root)
    mask_action_dir_list = [os.path.join(mask_root, action) for action in action_list]
    save_dir = '/data/shufan/shufan/mmaction2/data/kinetics400/conceptlist_classwise_for_inference'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for mask_action_dir in mask_action_dir_list:
        concept_list = os.listdir(mask_action_dir)
        save_path = os.path.join(save_dir, '{}.txt'.format(os.path.basename(mask_action_dir)))
        write_data = []
        for concept in concept_list:
            concept_dir = os.path.join(mask_action_dir, concept)
            concept_name = os.path.basename(concept_dir).replace(' ', '_')
            video_list = os.listdir(concept_dir)
            video_path_list = [os.path.join(concept_dir, video) for video in video_list]
            write_data += ['{} 0\n'.format(video_path) for video_path in video_path_list]
        save_txt(save_path, write_data)
