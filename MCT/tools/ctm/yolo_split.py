import os
import os.path as osp
if __name__ == "__main__":
    data_root = "/data/shufan/shufan/mmaction2/data/kinetics400/train_256"
    action_list = sorted(os.listdir(data_root))
    for action in action_list:
        action_path = osp.join(data_root, action)
        os.system('python /data/shufan/shufan/yolov5/detect.py --weights /data/shufan/shufan/yolov5/weights/yolov5s.pt --source {} --project /data/shufan/shufan/mmaction2/data/kinetics400/train_256_yolov5s_mask --exist-ok --half'.format(action_path))