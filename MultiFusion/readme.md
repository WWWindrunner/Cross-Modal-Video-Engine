Implementation of Composed Video Retrieval Task.

# MultiFusion

Implementation of Composed Video Retrieval Task.

## Environments

python = 3.8.15
pytorch = 1.9.1
cuda = 11.5
cuda_driver = 495.44
numpy = 1.23.5
tqdm
clip

## Dataset

TBD.

## Files in the folder:
(1) dataset/: videos datasebase
        modified_dataset/: our CVR dataset.
        videos/: videos that we need to retrieval.
(2) logs/: outputs of our running code.
(3) scripts/: train, test, and inference shell codes.
(4) src/: code source
        combiner.py: combiner model.
        combiner_train.py: training code based on modified_dataset.
        data_utils.py: dataset class code.
        inference.py: inference code that you can input reference video and modified text to retrieval top-1 video.
        utils.py: some utils code.
        validate.py: test code based on modified_dataset.

## Inference

```shell
Training the code:
    cd scripts
    sh train.sh
Test the code:
    cd scripts
    sh test.sh
Inference:
    cd scripts
    sh inference.sh
```

