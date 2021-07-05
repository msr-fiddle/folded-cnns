# FoldedCNNs
This repository contains the code used in the ICML 2021 paper:
"Boosting the Throughput and Accelerator Utilization of Specialized CNN Inference Beyond Increasing Batch Size". 
This source code is available under the [MIT License](LICENSE.txt).

## Repository structure
* [config](config): Configuration files used to describe each dataset/model
* [data](data): Directory to which the CIFAR and NoScope datasets are to be saved
* [datasets](datasets): Implementations of the curriculum learning and
                        distillation datasets used in the paper
* [dockerfiles](dockerfiles): Dockerfiles used for building Docker containers
                              to be used when running training and inference
                              experiments
* [model_files](model_files): Paths to pretrained "teacher" model for
                              distillation in CIFAR-10.
* [models](models): PyTorch implementation of models
* [util](util): Utility scripts

## Downloading datasets
This section details the access to the datasets used in this repository

### Game-scraping workload
The dataset from the game-scraping workload described in Section 2
and used in the evaluation of the paper may be downloaded from this
URL: [https://figshare.com/s/71fd0b25dbed73183079](https://figshare.com/s/71fd0b25dbed73183079).

You can un-tar the dataset by running:
```bash
tar -xf game_data.tar
```
The resultant directory will be ~1.6 GB in size.

### NoScope
The NoScope videos can be downloaded from the project [repository](https://github.com/stanford-futuredata/noscope).
The code used by our project to split the dataset into training, validation, and test sets is located
in [util](util). You will need `cv2` version 3.4.1 installed. For example, to
generate the `noscope-coral` dataset, download the `coral-reef-long.mp4` video 
from the link above and run:
```bash
cd util
python3 noscope_video_save.py /path/to/coral-reef-long.mp4 noscope-coral-frames-all
python3 noscope_split.py coral noscope-coral-subset --infile /path/to/coral-reef-long.csv --indir noscope-coral-frames-all
mv noscope-coral-subset ../data
```
This will require that you have at least 100 GB of storage. You can remove the
`noscope-frames-all` directory after successfully splitting the dataset.

### CIFAR-10 and CIFAR-100
The CIFAR-10 and CIFAR-100 datasets will be downloaded using the torchvision CIFAR
[dataloader](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar).

## Software and Requirements
* NVIDIA V100 GPU (we tested on an AWS p3.2xlarge instance) or NVIDIA T4 GPU (we tested on an AWS g4dn.xlarge instance)
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* CUDA 10.2
* NVIDIA Driver 418.87.01
* Other requirements are satisfied by the [provided Dockerfile](dockerfiles/FoldDockerfile)

You are encouraged to set the absolute path to this repository, as well as
that to the un-tarred `game_data` above:
```bash
export FOLD_HOME=$(pwd)
export DATA_ROOT=/path/to/game_data
```

To build the Docker image used in evaluation, perform the following:
```bash
cd $FOLD_HOME/dockerfiles
docker build -t fold -f FoldDockerfile .
```

You can then launch a Docker container with this image via:
```bash
docker run -it --rm --gpus all --shm-size=1g --ulimit memlock=-1 \
       --ulimit stack=67108864 --privileged=true \
       -v ${FOLD_HOME}:/workspace/folding \
       -v ${DATA_ROOT}:/workspace/folding/data_root \
       fold:latest
```

You should find yourself in the `/workspace` directory with this repository
under `/workspace/folding`. Navigate to `/workspace/folding` for the remainder
of the steps.

## Training
The logic for training a FoldedCNN occurs in [train.py](train.py) and 
[fold_trainer.py](fold_trainer.py). `train.py` orchestrates the training of
many models, and `fold_trainer.py` trains a single model.

`train.py` is currently configured to begin running all training
experiments described in the paper. If you would like to run fewer training
runs, you can edit the `datasets_to_run` and `folds` variables in `__main__`.

To train, run:
```bash
python3 train.py savedir
```
This will print training status like the following for each dataset and fold
combination:
```
Epoch 0. train. Top-1=0.2070, Top-5=0.6214, Loss=2.1715:  20%|#####          | 687/3438 [00:06<00:18, 149.32it/s]
```
The accuracies and model checkpoints used in a particular run will be saved
under the `savedir` directory passed in to `train.py` above.

## Inference
Inference experiments can be performed using the [run_inference.sh](run_inference.sh)
script. By default, this will run 10 trials of all models, batch sizes, and
fold values considered in evaluation. To change which configurations are run,
edit [run_inference.sh](run_inference.sh).

You can run the script as follows:
```bash
# Pipe stderr to a file so as to surpress unrelated PyTorch warning about ffmpeg
./run_inference.sh results.csv 2> stderr.txt
```
This will write to stdout lines of the form:
```
Model,Trial,Fold,Batch Size,Mode,Throughput,FLOPs/sec
lol/goldnumber-fraction,1,1,1024,Original,YYY,ZZZ
```
where `YYY` and `ZZZ` are the throughput (in images/sec) and FLOPs/sec
achieved in this particular configuration.

These results will also be saved to `results.csv`, the file indicated in
the invocation of `run_inference.sh` above.

### Notes when running inference on a T4 GPU
The T4 GPU is [known](https://arxiv.org/abs/1903.07486) to face performance throttling issues due to
overheating. To avoid these events, when running experiments on the T4, we lock
the GPU clock frequency to avoid overheating (as suggested in a related NVIDIA
[repository](https://github.com/NVIDIA/cutlass/issues/154#issuecomment-745426099)).
These commands are not needed for the V100 experiments:
```bash
nvidia-smi -i 0 -pm 1
sudo nvidia-smi -lgc 900 -i 0
```

## Other contents of this repository
The FLOP count of each model used in inference evaluation is calculated
using [thop](https://github.com/Lyken17/pytorch-OpCounter), which is installed
in the provided Dockerfile. The [flop_count.sh](flop_count.sh) script can
be used to retrieve the FLOP count for models and fold values used in evaluation.
The results of running this script are also saved in [config/model_map.json](config/model_map.json).
Note that the results of this script show the number of FLOPs performed with one
input, but for FoldedCNNs, this corresponds to `f` images. For FoldedCNNs,
this therefore represents the number of FLOPs performed for `f` images, rather
than for one.

The script [ai.py](ai.py) can be used to calculate the arithmetic intensities
of original CNNs, FoldedCNNs, and CNNs transformed by EfficientNet-style
compound scaling.

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## License
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT license](LICENSE.txt).
