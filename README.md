Scene Text Recognition with<br/>Permuted Autoregressive Sequence Models








Darwin Bautista
 and Rowel Atienza

Electrical and Electronics Engineering Institute<br/>
University of the Philippines, Diliman

Method
 | Sample Results
 | Getting Started
 | CLIP-Enhanced PARSeq (optional)
 | Training
 | Evaluation
 | Tuning
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
