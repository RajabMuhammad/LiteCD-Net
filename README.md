# MSKD-Net: Knowledge Distillation with Multi-Scale Spatial Alignment for Efficient Remote Sensing Change Detection

PyTorch implementation of a lightweight change detection model that distills the Visual change Transformer (VcT) teacher into a compact MobileNetV2-based student.

This work builds upon **VcT** [Jiang et al., IEEE TGRS 2023] — [[arXiv](https://arxiv.org/abs/2310.11417)] [[IEEE](https://ieeexplore.ieee.org/document/10294300)] — which serves as the teacher network.

## Abstract

Vision Transformers achieve state-of-the-art accuracy in remote sensing change detection, but their computational overhead limits deployment on resource-constrained platforms. We propose a knowledge distillation framework that compresses the Visual change Transformer (VcT) into a lightweight MobileNetV2-based student. To prevent spatial detail loss during compression, we introduce a U-Net decoder with multi-scale skip connections that computes change features at four encoder scales. Distillation is performed at two levels feature-map alignment at native resolution and soft-logit matching supervised by a combined cross-entropy and Dice loss with a warmup schedule. On the LEVIR-CD benchmark, our 2.56M-parameter student achieves 92.9% mean F1-score, retaining 98.9% of the 12.6M-parameter teacher's accuracy at 4.9× compression. These results demonstrate that architectural preservation of spatial detail is the dominant factor in successful change detection distillation.

## Method Overview

- **Teacher:** Visual change Transformer (VcT), 12.6M parameters — a hybrid ResNet-18 + graph-convolution + transformer architecture (frozen during distillation).
- **Student:** MobileNetV2 backbone (ImageNet-pretrained) with a U-Net decoder, 2.56M parameters.
- **Multi-scale change features:** absolute differences of bi-temporal features at strides 4, 8, 16, and 32.
- **Progressive U-Net decoder:** 8×8 → 16×16 → 32×32 → 64×64 → 256×256 with skip connections preserving fine building edges.
- **Two-level knowledge distillation:**
  - Feature-level MSE at native 32×32 resolution.
  - Logit-level KL-divergence with temperature scaling (T = 4).
- **Phased training:** 10-epoch warmup (cross-entropy + Dice) before distillation activates.

## Results (LEVIR-CD test set)

| Metric | Teacher (VcT) | Student (Ours) |
|--------|---------------|----------------|
| mF1 (%) | 92.4 | 92.9 (0.5) |
| F1 — change (%) | 88.4 | 88.2 |
| Recall — change (%) | 84.4 | 81.2 |
| Precision — change (%) | 92.2 | 93.5 (1.3) |
| IoU — change (%) | 79.3 | 76.1 |
| Parameters (M) | 12.6 | 2.56 |

The student retains 98.9% of the teacher's mean F1 while using 4.9× fewer parameters.

## Requirements

```
Python 3.7+
pytorch 1.11.0
torchvision
einops 0.6.0
torch-scatter 2.0.9   # required by the teacher
scipy 1.7.3
matplotlib 3.5.3
```

## Train

The dataset path is set in `data_config.py`. Train the student with:

```bash
sh run_cd.sh
```

`run_cd.sh`:

```bash
gpus=0
checkpoint_root=checkpoints
data_name=LEVIR

img_size=256
batch_size=8
lr=0.01
max_epochs=200
net_G=Student_transformer   # student model
lr_policy=linear

split=train
split_val=val
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name} --lr ${lr}
```

> The teacher checkpoint must exist at `checkpoints/CD_Reliable_transformer_LEVIR_b8_lr0.01_train_val_200_linear/best_ckpt.pt` before training the student, since distillation reads its outputs.

## Evaluate

```bash
sh eval.sh
```

`eval.sh`:

```bash
gpus=0
data_name=LEVIR
net_G=Student_transformer   # student model
split=test
project_name=CD_Student_transformer_LEVIR_b8_lr0.01_train_val_200_linear
checkpoint_name=best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```

## Visualization

Compare ground truth, teacher, and student predictions side by side:

```bash
python visualize_teacher_vs_student.py --num_samples 4
```

Output is saved to `vis/teacher_vs_student.png`.

## Dataset Preparation

### Data structure

```
Change detection dataset with pixel-level binary labels
├─A       # images of t1 phase
├─B       # images of t2 phase
├─label   # label maps
└─list    # train.txt, val.txt, test.txt (each lists image names XXX.png)
```

### Processed datasets

* **LEVIR-CD (2.3GB):** [[DropBox](https://www.dropbox.com/scl/fi/uvp5q311jul5hzrvoivte/LEVIR-CD-256.zip?rlkey=3uahso53jvdmjfvw7fbotwb36&dl=0)]


## Acknowledgement

The teacher network and dataset processing build on the official VcT implementation:

```
@article{jiang2023vct,
  title={VcT: Visual change Transformer for Remote Sensing Image Change Detection},
  author={Jiang, Bo and Wang, Zitian and Wang, Xixi and Zhang, Ziyan and Chen, Lan and Wang, Xiao and Luo, Bin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023}
}
```

## License

Code is released for non-commercial and research purposes **only**.
