# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply
from ..utils import init_weights

from .CLIP import clip,model,simple_tokenizer
import torch
import string
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
padding = False

 #Leehakho
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class only_clip():

    def __init__(self):
        self._device = device
        self.CLIPmodel, self.CLIPpreprocess = clip.load('ViT-B/16')
        self.max_label_length = 25
        self.charset_train = string.digits + string.ascii_lowercase
        dic = simple_tokenizer.SimpleTokenizer(max_label_length= self.max_label_length, charset = self.charset_train)
        self.label_origin = dic.getLabelVocab()
        self.new = True
        
        self.load_features = True
        self.save = False

        if self.load_features:
            self.new = False
            # 파일에서 텐서를 불러오기
            features = torch.load('text_features_6000.pth').to(self._device)
            number = ["12000", "18000", "24000", "30000"]
            #number = []
            for num in number:
                print(num)
                temp = torch.load('text_features_' + num + '.pth').to(self._device)
                features = torch.cat((features, temp), axis=0)
            print(features.shape)
            self.text_features = features
            self.label = self.label_origin

        else:
            if padding:
                self.label = self.label_origin[:1]
                
            self.label = self.label_origin
            print(len(self.label))
            self.text_token = torch.cat([clip.tokenize(f"word {c}") for c in self.label]).to(self._device)

        if self.save:
            s = 0
            e = s + 30000
            if e > len(self.label):
                e = len(self.label)
            print(s,e)
            print(self.text_token.shape)
            self.text_token_new = self.text_token[s:e]   
            self.text_features = self.txtencode(self.text_token_new.to(self._device))
            torch.save(self.text_features, "text_features_sum_" + str(e) + ".pth")

    def clip_encode(self, img: torch.Tensor):
        #print(img.shape)
        img = F.interpolate(img, size=224)
        #print(img.shape)
        #img = F.interpolate(img.unsqueeze(0), size=(3,224,224), mode='bilinear', align_corners=False)
        #print(self.CLIPpreprocess(img))
        with torch.no_grad():
            #emb = self.CLIPmodel.visual(img)
            #print(img.shape)
            emb = self.CLIPmodel.visual(img.type(self.CLIPmodel.dtype))
        return emb
        #return self.encoder(img)

    def txtencode(self, text: torch.Tensor):

        #print(self.CLIPpreprocess(img))
        with torch.no_grad():
            emb = self.CLIPmodel.encode_text(text)
        return emb

    def decode(self, x: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):

        if self.new:
            self.text_features = self.txtencode(self.text_token.to(self._device))
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            self.new = False
        else:
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        #print(text_features.shape)
        tgt_list = []
        for image_features in x:
            image_features /= image_features.norm(dim=-1, keepdim=True)
            #print(image_features.shape)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            #print(similarity[0])
            values, indices = similarity.topk(1)

            #for value, index in zip(values, indices):
            #print(indices)
            tgt = self.label[indices]
            #print("index:", indices, " clip pred:", tgt)
            #tgt = self.tokenizer.encode(tgt, self._device)
            #print(tgt.shape)
            tgt_list.append(tgt)
        return tgt_list

    def forward(self, images: Tensor) -> Tensor:
        #print(images.shape, max_length)
        bs = images.shape[0]
        x, _ = self.clip_encode(images)


        # Query positions up to `num_steps`
        pos_queries = None

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        #tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        # No prior context, so input is just <bos>. We query all positions.
        tgt_out = self.decode(x,tgt_query=pos_queries)
        #print(tgt_out.shape) #torch.Size([64, 26, 384])
        #logits = self.head(tgt_out)
       
        return tgt_out

