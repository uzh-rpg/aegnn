# AEGNN: Asynchronous Event-based Graph Neural Networks

## Installation
Create and activate an [Anaconda](https://www.anaconda.com/) environment.
```
conda create -n aegnn python=3.8
conda activate aegnn
``` 

The code heavily depends on PyTorch and the [PyG](https://github.com/pyg-team/pytorch_geometric) framework, which is 
optimized only for GPUs supporting CUDA. For our implementation the CUDA version 10.2 is used. Install the project
requirements with:
```
pip3 install torch==1.9.1
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.1+cu102.html
pip3 install -r aegnn/requirements.txt
```

## Training Pipeline
We evaluated our approach on three datasets. [NCars](http://www.prophesee.ai/dataset-n-cars/), 
[NCaltech101](https://www.garrickorchard.com/datasets/n-caltech101) and 
[Prophesee Gen1 Automotive](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/).
Download them and extract them. By default, they are assumed to be in `/data/storage/`, this can be changed by setting
the `AEGNN_DATA_DIR` environment variable. 

### Pre-Processing
To efficiently train the graph neural networks, the event graph is generated offline during pre-processing. 
```
CUDA_VISIBLE_DEVICES=X python3 aegnn/scripts/preprocessing.py --dataset dataset --num-workers 16
```

### Training
We use the [PyTorch Lightning](https://www.pytorchlightning.ai/) for training and [WandB](https://wandb.ai/) for
logging. By default, the logs are stored in `/data/logs/`, this can be changed by setting the `AEGNN_LOG_DIR` 
environment variable. To run our training pipeline:
```
python3 aegnn/scripts/train.py graph_res --task recognition --dataset dataset --gpu X --batch-size X --dim 3
```
with tasks `recognition` or `detection`. A list of configuration arguments can be found by calling the `--help` flag. 
To evaluate the detection pipeline, compute the mAP score on the whole test dataset by running: 
```
python3 aegnn/evaluation/map_search.py model --dataset dataset --device X
```

## Asynchronous & Sparse Pipeline
The code allows to make any graph-based convolutional model asynchronous & sparse, with a simple command and without 
the need to change the model's definition or forward function.
```
>>> import aegnn
>>> model = GraphConvModel()
>>> model = aegnn.asyncronous.make_model_asynchronous(model, **kwargs)
```
We support all graph convolutional layers, max pooling, linear layers and more. As each layer is independently 
transformed to work asynchronously and sparsely, if there is a layer, that we do not support, its dense equivalent 
is used instead. For evaluation, we support automatic flops and runtime analysis. As an example, we refer to:
```
python3 aegnn/evaluation/flops.py --device X
```