Getting Started
Installation

Requires Python >= 3.8 and PyTorch >= 1.10 (until 1.13).

# 1) Choose your torch platform build (cpu/cu116/cu117/rocm5.2)
platform=cpu
make torch-${platform}

# 2) Install core + train + test deps
pip install -r requirements/core.${platform}.txt -e .[train,test]

# (Optional) For CLIP encoder support
pip install git+https://github.com/openai/CLIP.git

Datasets

Download the datasets
 from the following links:

LMDB archives
 for MJSynth, SynthText, IIIT5k, SVT, SVTP, IC13, IC15, CUTE80, ArT, RCTW17, ReCTS, LSVT, MLT19, COCO-Text, and Uber-Text.

LMDB archives
 for TextOCR and OpenVINO.

Pretrained via Torch Hub (baseline PARSeq)
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img = Image.open('/path/to/image.png').convert('RGB')
img = img_transform(img).unsqueeze(0)  # (B, C, H, W)

logits = parseq(img)
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label =', label[0])

CLIP-Enhanced PARSeq (optional)

What is this?
We add an option to use a CLIP ViT image encoder (e.g., ViT-B/32) as the visual backbone of PARSeq. The decoder, tokenizer, and PARSeq’s permuted AR training remain the same; only the image encoder is swapped to leverage CLIP’s robust visual features.

Key points

Encoder: CLIP ViT (default: ViT-B/32).

Normalization: CLIP mean/std; CLIP resize/crop or letterbox (depending on your implementation).

Training: either freeze the CLIP encoder for a few epochs then unfreeze, or fine-tune end-to-end from the start.

Quick commands (pick the one that matches your repo setup)

# A) If you have a ready experiment config
./train.py +experiment=parseq-clip-b32 trainer.accelerator=gpu trainer.precision=16

# B) Or explicit overrides (example names; adjust to your actual arg keys)
./train.py model.encoder=clip_vit_b32 data.normalize=clip model.img_size=[224,224] \
           trainer.accelerator=gpu trainer.precision=16


Inference / Evaluation with a CLIP-PARSeq checkpoint

# Inference on a folder of images
./read.py /path/to/clip_parseq.ckpt --images demo_images/* \
          refine_iters:int=2 decode_ar:bool=false

# Evaluate on benchmarks
./test.py /path/to/clip_parseq.ckpt --cased --punctuation --batch_size 512 --num_workers 16

Training

You can override any configuration from the CLI (Hydra). See ./train.py --help.

# Finetune using pretrained weights (baseline)
./train.py pretrained=parseq-tiny

# Train a model variant/preconfigured experiment
./train.py +experiment=parseq-tiny

# Character set
./train.py charset=94_full             # or 36_lowercase / 62_mixed-case

# Dataset choice
./train.py dataset=real                # or synth

# Typical trainer flags
./train.py trainer.max_epochs=20 trainer.accelerator=gpu trainer.devices=2

Evaluation

test.py evaluates any trained checkpoint. You may pass PARSeq runtime params as param:type=value.

# Baseline evaluation (36-char set)
./test.py outputs/<model>/<ts>/checkpoints/last.ckpt

# Mixed-case + punctuation (94-char set)
./test.py outputs/<model>/<ts>/checkpoints/last.ckpt --cased --punctuation

# PARSeq NAR decoding with 2 refinement iters
./test.py outputs/<model>/<ts>/checkpoints/last.ckpt refine_iters:int=2 decode_ar:bool=false


Example output format

| Dataset   | # samples | Accuracy | 1 - NED | Confidence | Label Length |
| IIIT5k    |      3000 |    99.00 |   99.79 |      97.09 |         5.09 |
...
| Combined  |     7672  |    95.95 |   98.78 |      95.34 |         5.33 |

Tuning

We use Ray Tune
 for automated LR search. See ./tune.py --help.

./tune.py tune.num_samples=20
./tune.py +experiment=tune_abinet-lm
