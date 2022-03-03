# Vision-Language Pre-Training with Triple Contrastive Learning, CVPR 2022

This is the official PyTorch implementation of [TCL](https://arxiv.org/abs/2202.10401)

<img width="800" alt="image" src="https://user-images.githubusercontent.com/20442927/154851838-5297cc88-47d2-43f4-9602-ef29c63c479b.png">

### Requirements:
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install transformers==4.8.1
pip install timm==0.4.9
conda install ruamel_yaml
pip install opencv-python
pip install --upgrade Pillow
pip install einops
```

### Pre-training Datasets:
- [MSCOCO (2014)](https://cocodataset.org/#download)
- [Visual Genome (VG)](https://visualgenome.org/api/v0/api_home.html)
  - Download images of part1 and part2 and combine them together 
- Conceptual Captions (CC3M)
  - Download `Train_GCC-training.tsv` and `Validation_GCC-1.1.0-Validation.tsv` from [kaggle](https://www.kaggle.com/ad271828/conceptual-captions-dataset-train-and-validation)
  - Then use [img2dataset](https://github.com/rom1504/img2dataset) to download images from downloaed tsv files
  - [More details](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md)
- SBU Captions
  - Download url from [subcaptions](http://www.cs.virginia.edu/~vicente/sbucaptions/)
  - Then use [img2dataset](https://github.com/rom1504/img2dataset) to download images from url
- CC12M
  - Download [cc12m.tsv](https://github.com/google-research-datasets/conceptual-12m#download)
  - Then use [img2dataset](https://github.com/rom1504/img2dataset) to download images from the downloaed tsv file

### Downstream-task Datasets:
- [Flicker30k](https://www.kaggle.com/hsankesara/flickr-image-dataset)
- [VQA v2](https://visualqa.org/download.html)
- NLVR2
  - recommend to use [direct-image-download](https://github.com/lil-lab/nlvr/tree/master/nlvr2#direct-image-download)

### Json Files from Pre-training and Downstream Tasks:
- refer to [Download](https://github.com/salesforce/ALBEF#download) in [ALBEF](https://github.com/salesforce/ALBEF)
- you need to change the image path in json files according to your downloaded images

### Pre-trained checkpoint:
- [TCL_4M](https://drive.google.com/file/d/1rv77uqt1-YcYFPqavaFwccEgsO24gGsX/view?usp=sharing)

### Pre-training:
```
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env Pretrain.py \
--config ./configs/Pretrain.yaml \
--output_dir output/pretrain
```
### Downstream Tasks:
#### Image-Text Retrieval
```
# zero-shot coco 
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env Retrieval.py \
--config ./configs/Retrieval_coco.yaml \
--output_dir output/pretrain_e30_Retrieval_coco_zeroshot \
--checkpoint output/pretrain/checkpoint_29.pth \
--evaluate

# fine-tune flickr
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env Retrieval.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir output/pretrain_e30_Retrieval_flickr \
--checkpoint output/pretrain/checkpoint_29.pth

# fine-tune coco
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env Retrieval.py \
--config ./configs/Retrieval_coco.yaml \
--output_dir output/pretrain_e30_Retrieval_coco \
--checkpoint output/pretrain/checkpoint_29.pth

# zero-shot flickr 
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env Retrieval.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir output/pretrain_e30_Retrieval_flickr_zeroshot \
--checkpoint output/pretrain_e30_Retrieval_coco/checkpoint_best.pth \
--evaluate
```

#### VQA
```
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env VQA.py \
--config ./configs/VQA.yaml \
--output_dir output/pretrain_e30_vqa \
--checkpoint output/pretrain/checkpoint_29.pth
```
- how to evaluate and interpret the results(https://github.com/salesforce/ALBEF/issues/19)

#### Visual Entailment
```
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env VE.py \
--config ./configs/VE.yaml \
--output_dir output/pretrain_e30_VE \
--checkpoint output/pretrain/checkpoint_29.pth
```
- how to evaluate and interpret the results(https://github.com/salesforce/ALBEF/issues/19)

#### NLVR2
```
# pre-train nlvr
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env Pretrain_nlvr.py \
--config ./configs/NLVR_pretrain.yaml \
--output_dir output/pretrain_e30_NLVR_pretrain \
--checkpoint output/pretrain/checkpoint_29.pth

# fine-tune nlvr
python -m torch.distributed.launch --nproc_per_node=8 \
--use_env NLVR.py \
--config ./configs/NLVR.yaml \
--output_dir output/pretrain_e30_NLVR \
--checkpoint output/pretrain_e30_NLVR_pretrain/checkpoint_00.pth
```
- how to evaluate and interpret the results(https://github.com/salesforce/ALBEF/issues/19)

### Citation:
```
@article{yang2022vision,
  title={Vision-Language Pre-Training with Triple Contrastive Learning},
  author={Yang, Jinyu and Duan, Jiali and Tran, Son and Xu, Yi and Chanda, Sampath and Chen, Liqun and Zeng, Belinda and Chilimbi, Trishul and Huang, Junzhou},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2022}
}
```
Our code is largely borrowed from [ALBEF](https://github.com/salesforce/ALBEF)
