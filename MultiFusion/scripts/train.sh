cd src
CUDA_VISIBLE_DEVICES="0" python  combiner_train.py --dataset ComposedVideo  --save-best --save-training
