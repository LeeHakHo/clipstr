<div align="center">
Scene Text Recognition with<br/>Permuted Autoregressive Sequence Models








Darwin Bautista
 and Rowel Atienza

Electrical and Electronics Engineering Institute<br/>
University of the Philippines, Diliman

Method
 | Sample Results
 | Getting Started
 | FAQ
 | Training
 | Evaluation
 | Tuning
 | This Fork Notes
 | Citation

</div>

Scene Text Recognition (STR) models use language context to be more robust against noisy or corrupted images. Recent approaches like ABINet use a standalone or external Language Model (LM) for prediction refinement. In this work, we show that the external LM—which requires upfront allocation of dedicated compute capacity—is inefficient for STR due to its poor performance vs cost characteristics. We propose a more efficient approach using permuted autoregressive sequence (PARSeq) models. View our ECCV poster
 and presentation
 for a brief overview.

NOTE: P-S and P-Ti are shorthands for PARSeq-S and PARSeq-Ti, respectively.

Method tl;dr

Our main insight is that with an ensemble of autoregressive (AR) models, we could unify the current STR decoding methods (context-aware AR and context-free non-AR) and the bidirectional (cloze) refinement model:

<div align="center"><img src=".github/contexts-example.png" alt="Unified STR model" width="75%"/></div>

A single Transformer can realize different models by merely varying its attention mask. With the correct decoder parameterization, it can be trained with Permutation Language Modeling to enable inference for arbitrary output positions given arbitrary subsets of the input context. This arbitrary decoding characteristic results in a unified STR model—PARSeq—capable of context-free and context-aware inference, as well as iterative prediction refinement using bidirectional context without requiring a standalone language model. PARSeq can be considered an ensemble of AR models with shared architecture and weights:


NOTE: LayerNorm and Dropout layers are omitted. [B], [E], and [P] stand for beginning-of-sequence (BOS), end-of-sequence (EOS), and padding tokens, respectively. T = 25 results in 26 distinct position tokens. The position tokens both serve as query vectors and position embeddings for the input context. For [B], no position embedding is added. Attention masks are generated from the given permutations and are used only for the context-position attention. L<sub>ce</sub> pertains to the cross-entropy loss.

Sample Results
<div align="center">
Input Image	PARSeq-S<sub>A</sub>	ABINet	TRBA	ViTSTR-S	CRNN
<img src="demo_images/art-01107.jpg" alt="CHEWBACCA" width="128"/>	CHEWBACCA	CHEWBAGGA	CHEWBACCA	CHEWBACCA	CHEWUACCA
<img src="demo_images/coco-1166773.jpg" alt="Chevron" width="128"/>	Chevrol	Chevro_	Chevro_	Chevr__	Chevr__
<img src="demo_images/cute-184.jpg" alt="SALMON" height="128"/>	SALMON	SALMON	SALMON	SALMON	SA_MON
<img src="demo_images/ic13_word_256.png" alt="Verbandstoffe" width="128"/>	Verbandsteffe	Verbandsteffe	Verbandstelle	Verbandsteffe	Verbandsleffe
<img src="demo_images/ic15_word_26.png" alt="Kappa" width="128"/>	Kappa	Kappa	Kaspa	Kappa	Kaada
<img src="demo_images/uber-27491.jpg" alt="3rdAve" height="128"/>	3rdAve	3=-Ave	3rdAve	3rdAve	Coke

NOTE: Bold letters and underscores indicate wrong and missing character predictions, respectively.

</div>
Getting Started

This repository contains the reference implementation for PARSeq and reproduced models (collectively referred to as Scene Text Recognition Model Hub). See NOTICE for copyright information.
Majority of the code is licensed under the Apache License v2.0 (see LICENSE) while ABINet and CRNN sources are
released under the BSD and MIT licenses, respectively (see corresponding LICENSE files for details).

Demo

An interactive Gradio demo
 hosted at Hugging Face is available. The pretrained weights released here are used for the demo.

Installation

Requires Python >= 3.8 and PyTorch >= 1.10 (until 1.13). The default requirements files will install the latest versions of the dependencies (as of June 1, 2023).

# Use specific platform build. Other PyTorch 1.13 options: cu116, cu117, rocm5.2
platform=cpu
# Generate requirements files for specified PyTorch platform
make torch-${platform}
# Install the project and core + train + test dependencies. Subsets: [train,test,bench,tune]
pip install -r requirements/core.${platform}.txt -e .[train,test]

Updating dependency version pins
pip install pip-tools
make clean-reqs reqs  # Regenerate all the requirements files

Datasets

Download the datasets
 from the following links:

LMDB archives
 for MJSynth, SynthText, IIIT5k, SVT, SVTP, IC13, IC15, CUTE80, ArT, RCTW17, ReCTS, LSVT, MLT19, COCO-Text, and Uber-Text.

LMDB archives
 for TextOCR and OpenVINO.

Pretrained Models via Torch Hub

Available models are: abinet, crnn, trba, vitstr, parseq_tiny, and parseq.

import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img = Image.open('/path/to/image.png').convert('RGB')
# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img = img_transform(img).unsqueeze(0)

logits = parseq(img)
logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

# Greedy decoding
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))

Frequently Asked Questions

How do I train on a new language? See Issues #5
 and #9
.

Can you export to TorchScript or ONNX? Yes, see Issue #12
.

How do I test on my own dataset? See Issue #27
.

How do I finetune and/or create a custom dataset? See Issue #7
.

What is val_NED? See Issue #10
.

Training

The training script can train any supported model. You can override any configuration using the command line. Please refer to Hydra
 docs for more info about the syntax. Use ./train.py --help to see the default configuration.

What’s different in this fork (training)

Seed & reproducibility: Global seed fixed to 2023 (random, numpy, torch, cuda).

AMP & DDP defaults: If trainer.accelerator=gpu, mixed precision (fp16) on. If trainer.devices>1, DDP enabled with gradient_as_bucket_view=True.

find_unused_parameters on by default (safer; set to False in code if you’re sure all params are used).

Throughput scaling: If multi-GPU, val_check_interval and max_steps are scaled to keep effective frequency consistent.

Checkpointing: ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=3, save_last=True, filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}').

SWA: Enabled by default; starts at swa_epoch_start=0.75. SWA LR derived from base LR and warmup (warmup_pct).

Pretrained init: pretrained=<name> loads weights via helper; errors out early if mismatch.

PARSeq safety: When perm_mirrored=True, enforces even perm_num.

Logging: TensorBoardLogger writes under Hydra output dir (or checkpoint parent when resuming).

<details><summary>Sample commands for different training configurations</summary><p>
Finetune using pretrained weights
./train.py pretrained=parseq-tiny  # Not all experiments have pretrained weights

Train a model variant/preconfigured experiment

The base model configurations are in configs/model/, while variations are stored in configs/experiment/.

./train.py +experiment=parseq-tiny  # Some examples: abinet-sv, trbc

Specify the character set for training
./train.py charset=94_full  # Other options: 36_lowercase or 62_mixed-case. See configs/charset/

Specify the training dataset
./train.py dataset=real  # Other option: synth. See configs/dataset/

Change general model training parameters
./train.py model.img_size=[32, 128] model.max_label_length=25 model.batch_size=384

Change data-related training parameters
./train.py data.root_dir=data data.num_workers=2 data.augment=true

Change pytorch_lightning.Trainer parameters
./train.py trainer.max_epochs=20 trainer.accelerator=gpu trainer.devices=2


Note that you can pass any Trainer parameter
,
you just need to prefix it with + if it is not originally specified in configs/main.yaml.

Resume training from checkpoint (experimental)
./train.py +experiment=<model_exp> ckpt_path=outputs/<model>/<timestamp>/checkpoints/<checkpoint>.ckpt

</p></details>
Evaluation

The test script, test.py, can be used to evaluate any model trained with this project. For more info, see ./test.py --help.

What’s different in this fork (evaluation)

New CLI flags:
--data_root (default data), --batch_size (default 512), --num_workers (default 16), --device (default cuda).

Charset presets: --charset {eng,kor,merge,chinese,eng_cn} (+ --cased, --punctuation).

Dataset switch: --data eng (classic English benchmarks), --data chinese, --data eng_cn (English + Chinese).

Rotation robustness: --rotation <deg> (CCW) applies inside the datamodule.

Quick inference: --inference (experimental) runs a small folder; update the hardcoded root in the script (default: demo_images/sample_merge/).

Reports: Aggregates Accuracy / 1-NED / Confidence / Label length per dataset and combined; writes a markdown-style table to <checkpoint>.log.txt.

Seed: Fixed to 2023 for deterministic evaluation.

PARSeq runtime parameters can be passed using the format param:type=value. For example, PARSeq NAR decoding can be invoked via:

./test.py parseq.ckpt refine_iters:int=2 decode_ar:bool=false

<details><summary>Sample commands for reproducing results</summary><p>
Lowercase alphanumeric comparison on benchmark datasets (Table 6)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt  # or use the released weights: ./test.py pretrained=parseq


Sample output:

Dataset	# samples	Accuracy	1 - NED	Confidence	Label Length
IIIT5k	3000	99.00	99.79	97.09	5.09
SVT	647	97.84	99.54	95.87	5.86
IC13_1015	1015	98.13	99.43	97.19	5.31
IC15_2077	2077	89.22	96.43	91.91	5.33
SVTP	645	96.90	99.36	94.37	5.86
CUTE80	288	98.61	99.80	96.43	5.53
Combined	7672	95.95	98.78	95.34	5.33
Benchmark using different evaluation character sets (Table 4)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt                      # lowercase alphanumeric (36-character set)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased             # mixed-case alphanumeric (62-character set)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation  # mixed-case alphanumeric + punctuation (94-character set)

Lowercase alphanumeric comparison on more challenging datasets (Table 5)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --new

Benchmark Model Compute Requirements (Figure 5)
./bench.py model=parseq model.decode_ar=false model.refine_iters=3

<torch.utils.benchmark.utils.common.Measurement object at 0x...>
model(x)
  Median: 14.87 ms
  IQR:    0.33 ms (14.78 to 15.12)
  7 measurements, 10 runs per measurement, 1 thread
| module                | #parameters   | #flops   | #activations   |
|:----------------------|:--------------|:---------|:---------------|
| model                 | 23.833M       | 3.255G   | 8.214M         |
|  encoder              |  21.381M      |  2.88G   |  7.127M        |
|  decoder              |  2.368M       |  0.371G  |  1.078M        |
|  head                 |  36.575K      |  3.794M  |  9.88K         |
|  text_embed.embedding |  37.248K      |  0       |  0             |

Latency Measurements vs Output Label Length (Appendix I)
./bench.py model=parseq model.decode_ar=false model.refine_iters=3 +range=true

Orientation robustness benchmark (Appendix J)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation          # no rotation
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation --rotation 90
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation --rotation 180
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation --rotation 270

Using trained models to read text from images (Appendix L)
./read.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --images demo_images/*  # Or use ./read.py pretrained=parseq
# use NAR decoding + 2 refinement iterations for PARSeq
./read.py pretrained=parseq refine_iters:int=2 decode_ar:bool=false --images demo_images/*

</p></details>
Tuning

We use Ray Tune
 for automated parameter tuning of the learning rate. See ./tune.py --help. Extend tune.py to support tuning of other hyperparameters.

./tune.py tune.num_samples=20  # find optimum LR for PARSeq's default config using 20 trials
./tune.py +experiment=tune_abinet-lm  # find the optimum learning rate for ABINet's language model

This Fork (Implementation Notes)

This repository extends the original PARSeq with multi-GPU friendly defaults, SWA, deterministic runs, multilingual evaluation, and quality-of-life CLI switches.

Training

Seed fixed at 2023; AMP on GPU; DDP with gradient_as_bucket_view=True.

Default find_unused_parameters=True (toggle in code for speed if not needed).

Scales validation/checkpoint frequency for multi-GPU to maintain effective cadence.

SWA enabled with derived LR; robust checkpoint naming incl. val_accuracy & val_NED.

Evaluation

New flags: --data_root, --batch_size, --num_workers, --device.

Charset presets: eng, kor, merge (alnum + Hangul), chinese, eng_cn (+ --cased, --punctuation).

Dataset bundles: eng, chinese, eng_cn.

Rotation stress test: --rotation.

Quick --inference mode (update the sample folder path in the script).

Outputs a combined markdown report to <checkpoint>.log.txt.

Citation
@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  month={10},
  year={2022},
  publisher={Springer Nature Switzerland},
  address={Cham},
  doi={10.1007/978-3-031-19815-1_11},
  url={https://doi.org/10.1007/978-3-031-19815-1_11}
}
