#!/usr/bin/env bash

data_root=/data/shufan/shufan/mmaction2/data/kinetics400/train_256
cd /data/shufan/shufan/yolov5
for source_dir in ${data_root}/*
do
    echo ${source_dir}
    python detect.py --weights /data/shufan/shufan/yolov5/weights/yolov5s.pt --source ${source_dir} --project /data/shufan/shufan/mmaction2/data/kinetics400/train_256_yolov5s_mask --exist-ok --half
    #python detect.py --weights /data1/shufan/yolov5/weights/yolov5m.pt --source ${source_dir} --project /data1/shufan/mmaction2/data/kinetics400/yolosample/val_256_yolov5m_mask --exist-ok --half
    #python detect.py --weights /data1/shufan/yolov5/weights/yolov5l.pt --source ${source_dir} --project /data1/shufan/mmaction2/data/kinetics400/yolosample/val_256_yolov5l_mask --exist-ok --half
    #python detect.py --weights /data1/shufan/yolov5/weights/yolov5x.pt --source ${source_dir} --project /data1/shufan/mmaction2/data/kinetics400/yolosample/val_256_yolov5x_mask --exist-ok --half
done