# CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations

This repository contains the PyTorch implementation of **CORE-MTL**, a representation-centric multi-task learning framework for robust visual multi-task learning.

CORE-MTL is motivated by the observation that negative transfer in multi-task learning is often caused not only by conflicting gradients, but also by entangled shared representations. Instead of only reweighting or projecting task gradients, CORE-MTL encourages the shared representation to be factorized into:

- a **semantic stream**, which captures task-relevant and stable scene structure;
- a **residual stream**, which absorbs nuisance variation such as texture, lighting, background, or appearance changes.

Task predictors use only the semantic stream. Reconstruction, independence regularization, and counterfactual feature augmentation are used during training to shape the representation.

![CORE-MTL pipeline](assets/core_pipeline.png)

## Paper

**CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations**  
Accepted to ICML 2026.

If you use this code, please cite the paper listed in the [Citation](#citation) section.

## Highlights

- Representation-centric multi-task learning framework.
- Semantic--residual factorization of shared visual features.
- Physical or generic grounding through reconstruction.
- CKA-based independence regularization between semantic and residual streams.
- Counterfactual feature augmentation by residual-code swapping.
- Experiments on NYUv2, Cityscapes, GTA5 -> Cityscapes, Cityscapes-C, Colored-Cityscapes, and CelebA.

## Environment

The code was developed with PyTorch and CUDA-enabled GPUs. A typical environment can be created as follows:

```bash
conda create -n core-mtl python=3.10 -y
conda activate core-mtl
```

Install PyTorch according to your CUDA version. For example, for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install numpy pandas pillow pyyaml tqdm torchmetrics kornia lpips
```

The experiments in the paper were implemented in PyTorch and run on NVIDIA A800 GPUs with 80GB memory. Each experiment uses a single GPU unless otherwise specified.

## Data Preparation

Datasets are **not included** in this repository. Please download and preprocess the datasets separately, then update the dataset paths in the corresponding YAML configuration files under:

```text
configs/resnet/
```

For full details about dataset preprocessing and experimental settings, please refer to the paper appendix:

- **Appendix F.1**: datasets and preprocessing;
- **Appendix F.2**: optimization settings, batch sizes, epochs, and data augmentation;
- **Appendix I**: Colored-Cityscapes stress test construction;
- **Appendix J**: Cityscapes-C per-corruption evaluation details.

### First impression of the preprocessing pipeline

Before running the code, the most important preprocessing assumptions are:

1. **NYUv2 and Cityscapes follow LibMTL/MTAN-style preprocessing.**  
   The code expects preprocessed `.npy` files for images and task labels rather than raw dataset folders.

2. **Dense prediction inputs are resized to fixed resolutions.**  
   NYUv2 uses `288 x 384`, while Cityscapes-style experiments use `128 x 256`.

3. **Invalid pixels are masked out for depth and surface-normal tasks.**  
   For NYUv2, invalid depth and normal pixels are ignored during both training and evaluation.

4. **Cityscapes depth labels are pseudo-depth maps derived from stereo disparity.**  
   The Cityscapes experiments use the same preprocessed package as LibMTL-style dense multi-task learning.

5. **GTA5 -> Cityscapes is a sim-to-real setting.**  
   GTA5 is used as the source training domain, while Cityscapes is used as the real target validation domain.

6. **Cityscapes-C is used for corruption robustness evaluation.**  
   The corrupted images are evaluated together with the clean Cityscapes labels and depth maps.

7. **CelebA uses the official attribute annotations and partition file.**  
   Images are resized to `128 x 128`, and the number of attributes can be controlled in the configuration file.

### NYUv2

The NYUv2 experiment follows the LibMTL/MTAN preprocessing protocol.

The paper uses:

- 795 training images;
- 654 validation images;
- three tasks:
  - 13-class semantic segmentation;
  - depth estimation;
  - surface normal prediction;
- input resolution: `288 x 384`.

Expected structure:

```text
<NYUv2_ROOT>/
├── train/
│   ├── image/*.npy
│   ├── label/*.npy
│   ├── depth/*.npy
│   └── normal/*.npy
└── val/
    ├── image/*.npy
    ├── label/*.npy
    ├── depth/*.npy
    └── normal/*.npy
```

Important preprocessing and evaluation details:

- invalid depth and normal pixels are masked out;
- depth is evaluated using absolute error and relative error;
- no additional log transform or clipping is applied for depth evaluation;
- normal estimation is evaluated using mean angular error, median angular error, and accuracy under angular thresholds.

Set the dataset path in:

```yaml
configs/resnet/1causal_nyu.yaml
```

### Cityscapes

The Cityscapes in-distribution experiment follows the LibMTL-style preprocessed Cityscapes setting.

The paper uses:

- 2975 training images;
- 500 validation images;
- two tasks:
  - 7-class semantic segmentation;
  - monocular depth estimation;
- input resolution: `128 x 256`.

Expected structure:

```text
<Cityscapes_ROOT>/
├── train/
│   ├── image/*.npy
│   ├── label/*.npy
│   └── depth/*.npy
└── val/
    ├── image/*.npy
    ├── label/*.npy
    └── depth/*.npy
```

Important preprocessing and evaluation details:

- images and labels are stored as preprocessed NumPy arrays;
- depth labels are pseudo-depth maps derived from stereo disparity;
- segmentation is evaluated by mIoU and pixel accuracy;
- depth is evaluated by absolute error and relative error.

Set the dataset path in:

```yaml
configs/resnet/2causal_cityscapes.yaml
```

### GTA5 -> Cityscapes

The GTA5 -> Cityscapes experiment evaluates sim-to-real out-of-distribution generalization.

The source domain is GTA5, and the target validation domain is Cityscapes.

Expected GTA5 structure:

```text
<GTA5_ROOT>/
├── images/*.png
├── labels/*.png
└── depth/*.npy
```

Expected Cityscapes target validation structure follows the preprocessed Cityscapes format described above.

Important preprocessing details:

- GTA5 is used for source-domain training;
- Cityscapes is used for target-domain validation;
- an optional GTA5 validation set can be used for source-domain validation;
- the experiment uses the same `128 x 256` image size as the Cityscapes setting;
- random horizontal flipping is used as the primary augmentation.

Update the paths in:

```yaml
configs/resnet/3gta2cityscapes.yaml
```

### Cityscapes-C

Cityscapes-C is used to evaluate robustness under common corruptions.

The model is trained on clean Cityscapes and evaluated on corrupted Cityscapes images. The paper reports results averaged over several corruptions and also provides per-corruption results.

Expected corrupted-image structure:

```text
<CityscapesC_ROOT>/
└── <corruption>/<severity>/<city>/*.png
```

The clean Cityscapes labels and depth maps are still required for evaluation:

```text
<CleanCityscapes_ROOT>/
└── val/
    ├── label/*.npy
    └── depth/*.npy
```

Important evaluation details:

- corrupted images are used as inputs;
- clean Cityscapes labels and depth maps are used as ground truth;
- the paper reports results on corruptions such as Contrast, Defocus Blur, Fog, and Gaussian Noise;
- the averaged Cityscapes-C robustness result is reported in the main paper, while per-corruption results are provided in Appendix J.

Update the paths in:

```yaml
configs/resnet/4cityscapes_c.yaml
```

### Colored-Cityscapes

Colored-Cityscapes is a stress test designed to evaluate whether the model relies on residual appearance shortcuts.

The construction is described in Appendix I of the paper.

The key idea is:

- in the source distribution, each semantic class is assigned a fixed artificial color;
- in the target distribution, the class-color mapping is permuted;
- semantic labels and depth labels remain unchanged;
- only the appearance shortcut is changed.

This benchmark is used to test whether the model can reduce appearance leakage into the prediction stream.

### CelebA

The CelebA experiment evaluates scalability with an increasing number of binary attribute prediction tasks.

Expected structure:

```text
<CelebA_ROOT>/
├── img_align_celeba/
├── list_attr_celeba.txt
└── list_eval_partition.txt
```

Important preprocessing details:

- images are resized to `128 x 128`;
- the official attribute annotations are used;
- the official evaluation partition file is used;
- random horizontal flipping and mild color jitter are used during training;
- the number of attributes can be changed in the configuration file.

Update the dataset path in:

```yaml
configs/resnet/5celeba.yaml
```

To change the number of CelebA attributes, edit:

```yaml
num_attributes: 10
```

## Training

Before running any experiment, edit the dataset paths in the relevant YAML configuration file.

### NYUv2

```bash
python main.py --config configs/resnet/1causal_nyu.yaml
```

### Cityscapes

```bash
python main.py --config configs/resnet/2causal_cityscapes.yaml
```

### GTA5 -> Cityscapes

```bash
python main.py --config configs/resnet/3gta2cityscapes.yaml
```

### CelebA

```bash
python celeba_main.py --config configs/resnet/5celeba.yaml
```

## Main Configuration Options

Important configuration fields include:

```yaml
data:
  type: nyuv2
  dataset_path: path/to/dataset
  batch_size: 16
  num_workers: 4
  img_size: [288, 384]

model:
  encoder_name: resnet50
  pretrained: true
  latent_dim_s: 512
  latent_dim_p: 256

training:
  seed: 2024
  epochs: 100
  optimizer: AdamW
  learning_rate: 0.0002
  weight_decay: 0.0001

losses:
  lambda_independence: 1.0
```

Commonly adjusted fields:

- `data.dataset_path`: dataset location;
- `data.train_dataset_path`: source training dataset location for domain-transfer experiments;
- `data.val_dataset_path`: target validation dataset location;
- `data.batch_size`: batch size;
- `model.latent_dim_s`: semantic-stream latent dimension;
- `model.latent_dim_p`: residual-stream latent dimension;
- `training.epochs`: number of training epochs;
- `training.learning_rate`: initial learning rate;
- `losses.lambda_independence`: CKA independence weight;
- `training.cfa.lambda_cfa`: counterfactual feature augmentation weight.

## Method Summary

CORE-MTL contains three main training-time mechanisms.

### 1. Dual-stream representation

The encoder output is split into a semantic stream and a residual stream.

The semantic stream is used by task heads for prediction. The residual stream is encouraged to absorb nuisance variation such as texture, illumination, and background appearance.

### 2. Grounded reconstruction

The model uses reconstruction to ground the semantic--residual factorization.

For dense scene understanding tasks, the hard-grounding version uses a physics-inspired reconstruction decoder. The semantic stream is encouraged to capture geometric structure, while the residual stream captures photometric factors such as albedo and lighting.

For settings where explicit physical structure is unavailable, a generic convolutional decoder can be used for soft grounding.

### 3. Independence and counterfactual augmentation

CORE-MTL uses Linear CKA to reduce statistical dependence between the semantic and residual streams.

It also uses counterfactual feature augmentation by swapping residual codes within a batch. The synthesized counterfactual samples preserve semantic content while changing residual appearance, encouraging the semantic stream to remain stable under nuisance changes.

The reconstruction and counterfactual modules are used to shape the representation during training. At inference time, predictions are made from the semantic stream.

## Outputs

Dense prediction experiments save outputs under:

```text
runs/<timestamp>/
├── checkpoints/
├── visualizations/
└── log files
```

CelebA experiments save outputs under:

```text
runs_celeba/<timestamp>/
├── checkpoints/
├── visualizations/
└── log files
```

The best checkpoint is saved as:

```text
model_best.pth.tar
```

## Reproducibility Notes

- The paper uses AdamW with an initial learning rate of `2e-4` and weight decay `1e-4`.
- A cosine annealing learning-rate schedule is used.
- Warmup is used at the beginning of training.
- NYUv2, Cityscapes, GTA5 -> Cityscapes, and Cityscapes-C are trained for 100 epochs.
- CelebA is trained for 50 epochs.
- The exact batch sizes, loss weights, and augmentation details are provided in Appendix F of the paper.
- Dataset preprocessing should match the expected formats described above.
- If you encounter an out-of-memory error, reduce `batch_size`, `latent_dim_s`, or `latent_dim_p` in the configuration file.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{wu2026coremtl,
  title     = {CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations},
  author    = {Wu, Chengfeng and Wang, Jingge and Wu, Yanru and Zou, Tao},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## Acknowledgement

This implementation builds on common dense multi-task learning protocols and preprocessing conventions used in LibMTL-style benchmarks. We thank the authors of the public datasets and benchmark tools used in this work.
