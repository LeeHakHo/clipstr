import math
from functools import partial
from typing import Sequence, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import cat as cat

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import DecoderLayer, Decoder, TokenEmbedding

from .CLIP import clip, simple_tokenizer
from .simclr import SimCLR
import torch

import random
import sys
device = "cuda" if torch.cuda.is_available() else "cpu"

class PARCLIP(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        self.embed_dim = embed_dim
        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        #print( len(self.tokenizer) - 2) #37
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

        self.CLIPmodel, _ = clip.load('ViT-B/16')
        self.text_projection = self.CLIPmodel.text_projection
        self.text_prompt = nn.Parameter(torch.randn([1, 4, 512]))

        # 모델 파라미터 고정하기
        for param in self.CLIPmodel.parameters():
            #param.requires_grad = False
            if param.dtype == torch.float16 or param.dtype == torch.half:
                param.data = param.data.to(torch.float32)
        self.charset_train = charset_train

        # self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
        #                        mlp_ratio=enc_mlp_ratio)

        dic = simple_tokenizer.SimpleTokenizer(max_label_length= self.max_label_length, charset = self.charset_train)
        self.label = dic.getLabelVocab() #label 가져오기
        
        #Leehakho
        self.new = True
        self.padding = False #inference에 padding값을 넣을지
        self.load_features = False #저장된 text encode를 가져올지
        self.seperate = False #encode할때 text를 나눠서 하기
        self.text_pmt = True #text_prompt 사용 여부
        self.load = True #text prompt를 위한 unnormalized feature를 load할지 여부
        self.save = False #text prompt에서 encoding한 값을 저장할건지 여부
        self.contrastive = True #contrastive 사용 여부

    #def no_weight_decay(self):
    #    param_names = {'text_embed.embedding.weight', 'pos_queries'}
    #    enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
    #    return param_names.union(enc_param_names)

    def clip_encode(self, img: torch.Tensor):
        #img = F.interpolate(img, size=224)
        #with torch.no_grad():
        emb = self.CLIPmodel.encode_image(img)
        return emb

    #def encode(self, img: torch.Tensor):
    #    return self.encoder(img)

    def txtencode(self, text: torch.Tensor, normalize):
        with torch.no_grad():
            emb = self.CLIPmodel.encode_text(text, normalize)
        return emb
    

    def cat_prompt(self, text_features, prompt):
        prompt = prompt.expand(text_features.shape[0],-1,-1)
        #sos token 다음에 promt를 추가 sos+prompt+text
        text_features = cat((self.text_features[:,:0,:], prompt, self.text_features[:,1:74,:]), dim=1)
        return text_features
    
    @torch.jit.script
    def dotProduct(image, text):
        return (100.0 * image @ text.T)
    
    def decode(self, tgt: torch.Tensor, x: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None, GT: Optional[Tensor] = None):
        if self.new:
            self.CLIPmodel =self.CLIPmodel.to(self._device)
            if self.contrastive:
                self.simclr = SimCLR(self._device)
                self.criterion = torch.nn.CrossEntropyLoss().to(self._device)
            if self.text_pmt:
            
                self.label = self.label#[200000:]
                print(len(self.label))
                text_token =[]
                for l in self.label:
                    a = []
                    a.append(l)
                    text_token.append(cat([clip.tokenize(f"{c}") for c in a]).to(self._device))
                self.text_token = text_token
                if self.load:
                    self.text_features_tensor = torch.load('normal_seperate.pth',  map_location=self._device)
                    print("load features", self.text_features_tensor.shape)
                else:
                    text_features_list = []
                    i=0
                    for tokenized_text in text_token:
                        text_feature= self.txtencode(tokenized_text, normalize = True)
                        i += 1
                        sys.stdout.write(" words encoded" + "\r" + str(i))
                        sys.stdout.flush()
                        text_features_list.append(text_feature)

                    self.text_features_tensor = cat(text_features_list, dim= 0)
                    del text_features_list
                    del text_token
                    
                    if self.save:
                        torch.save(self.text_features_tensor, 'normal_seperate.pth')
                        print("saved tensor ", self.text_features_tensor.shape)
            elif self.load_features:
                self.text_features = torch.load('real_seperate.pth').to(self._device)
                self.text_features = cat([self.txtencode(tokenized_text) for tokenized_text in text_token]).to(device)
                self.text_features = self.text_features.to(self._device)
                self.text_features = self.text_features.to(torch.float32)
            else:
                if self.seperate:
                    print(len(self.label))
                    self.text_token =[]
                    for l in self.label:
                        a = []
                        a.append(l)
                        self.text_token.append(cat([clip.tokenize(f"word {c}") for c in a]).to(self._device))
                    self.text_token = text_token
                    self.text_features_tensor = cat([self.txtencode(tokenized_text) for tokenized_text in text_token]).to(device)
                else:
                    self.label = random.sample(self.label, 3000)

                    if self.padding:
                        self.label = self.label[:1]
                    
                    self.text_token = cat([clip.tokenize(f"word {c}") for c in self.label])
            self.new = False

        if self.text_pmt:
            text_features = text_features_tensor = self.text_features_tensor.to(torch.float32)
        
        clip_pred =[]
        candidate_features = []
        candidate_label = []
        for image_features in x:
            with torch.no_grad():
                text_features /= text_features.norm(dim=-1, keepdim=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            _, indices = similarity.topk(4)
            indx = indices[0]

            clip_pred.append(self.text_token[indx])

            if self.contrastive:
                candidate_features.append([text_features_tensor[idx] for idx in indices])
                candidate_label.append([self.label[i] for i in indices])
        clip_pred = cat(clip_pred, dim=0).to(self._device)

        text_features_list = []
        for tokenized_text in clip_pred:
            text_features_list.append(self.txtencode(tokenized_text.unsqueeze(0), normalize = False))
        tgt = cat(text_features_list, dim= 0)
        
        if self.text_pmt:
            text_f = tgt.to(self._device)
            prompt = self.text_prompt
            prompt = prompt.expand(text_f.shape[0],-1,-1).to(self._device)
            #sos token 다음에 promt를 추가 sos+prompt+text
            tgt = cat((text_f[:,:0,:], prompt, text_f[:,1:74,:]), dim=1)

        _, N, L = tgt.shape
        tgt_emb = self.dropout(tgt)


        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask), candidate_features, candidate_label

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        x, memory = self.clip_encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)


        # No prior context, so input is just <bos>. We query all positions.
        tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
        tgt_out, _, _ = self.decode(tgt_in, x, memory,tgt_query=pos_queries)
        logits = self.head(tgt_out)
  
        return logits

    def write_unique_strings_to_file(self, strings):
        existing_strings = set()
        file_path = "/home/ohh/PycharmProject/PARCLIP/labels.txt"
        try:
            with open(file_path, 'r') as file:
                existing_strings = set(file.read().splitlines())
        except FileNotFoundError:
            pass

        unique_strings = set(strings) - existing_strings  # 중복 제거
        with open(file_path, 'a') as file:
            for string in unique_strings:
                file.write(string + '\n')

    def training_step(self, batch, batch_idx, max_length: Optional[int] = None) -> STEP_OUTPUT:
        images, labels = batch

        #GT 개수 뽑아보기
        #self.write_unique_strings_to_file(labels)

        tgt = self.tokenizer.encode(labels, self._device)
        gt = tgt[:, 1:]

        loss = 0
        loss_numel = 0
        n = (gt != self.pad_id).sum().item()
        max_len = tgt.shape[1] -2 # exclude <eos> from count


        #forward
        max_length = max_len
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        x, memory = self.clip_encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # No prior context, so input is just <bos>. We query all positions.
        tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
        tgt_out, candidate_features, candidate_labels = self.decode(tgt_in, x, memory,tgt_query=pos_queries)
        out = self.head(tgt_out)



        logits = out.flatten(end_dim=1)
        loss = loss + (n * F.cross_entropy(logits, gt.flatten(), ignore_index=self.pad_id))
        loss_numel = loss_numel + n

        if self.contrastive:    
            #linear layer pred candidate
            probs = out.softmax(-1)
            preds, _ = self.tokenizer.decode(probs)
            idex = 0
            for pred in preds:
                candidate_labels[idex].append(pred)
                pred_token = clip.tokenize(f"{pred}").to(self._device)
                pred_feature = self.txtencode(pred_token, normalize = True)
                pred_feature = pred_feature.squeeze(0)
                candidate_features[idex].append(pred_feature)
                idex += 1


            #Clip candidate
            idex = 0
            temp = []
            for label, candidate in zip(labels, candidate_labels):
                if label in candidate:
                    del candidate_features[idex][candidate.index(label)]
                else:
                    del candidate_features[idex][-2]
                temp.append(torch.stack(candidate_features[idex], dim = 0).unsqueeze(0))
                idex += 1

            candidate_features = cat(temp,dim=0).to(self._device)
            #positive pair
            label_token = cat([clip.tokenize(f"{c}") for c in labels]).to(self._device)
            label_f  = torch.cat([self.txtencode(tokenized_text.unsqueeze(0), normalize = True) for tokenized_text in label_token])


            x, con_labels = self.simclr.my_loss(x, candidate_features, label_f)
            con_loss = self.criterion(x, con_labels).to(self._device)

        loss /= loss_numel
        if self.contrastive:
            total_loss = 0.995 * loss + 0.005 * con_loss
        else:
            total_loss = loss

        self.log('loss', total_loss)
        return total_loss
