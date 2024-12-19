# DM-Adapter: Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval


## Highlights

The goal of this work is to enhance implicit fine-grained knowledge transferring,  offering the best trade-off between performance and parameter efficiency.

## Usage
### Requirements
we use single NVIDIA 4090 24G GPU for training and evaluation. 
```
pytorch 1.12.1
torchvision 0.13.1
prettytable
easydict
```

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```


## Training

```python train.py \
--name baseline \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+aux' \
--num_epoch 60 \
--root_dir '.../dataset_reid' \
--lr 3e-4 \
--num_experts 6 \
--topk 2 \
--reduction 8
```

## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```

## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [IRRA](https://github.com/anosorae/IRRA) and [Mixtral](https://github.com/mistralai/mistral-inference). We sincerely appreciate for their contributions.