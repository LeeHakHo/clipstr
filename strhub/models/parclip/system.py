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

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import DecoderLayer, Decoder, Encoder, TokenEmbedding

from .CLIP import clip,model,simple_tokenizer
from .simclr import SimCLR
import torch

import matplotlib.pyplot as plt
import random
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
        

        self.new = True #첫번째 batch에서 encode를 한번만 하기 위한변수
        #Leehakho
        self.padding = False #inference에 padding값을 넣을지
        self.load_features = False #저장된 text encode를 가져올지
        self.seperate = False #encode할때 text를 나눠서 하기
        self.text_pmt = True #text_prompt 사용 여부
        self.load = False #text prompt를 위한 unnormalized feature를 load할지 여부
        self.contrastive = False

        if self.contrastive:
            self.simclr = SimCLR(self._device)
        if self.load_features:
            # 파일에서 텐서를 불러오기
            self.text_features = torch.load('real_seperate.pth').to(self._device)
            #number = ["60000", "87837"]
            #for num in number:
            #    temp = torch.load('text_features_new_' + num + '.pth').to(self._device)
            #    features = torch.cat((features, temp), axis=0)
            #self.tm = self.text_features
        elif self.text_pmt:
            self.text_prompt = nn.Parameter(torch.randn([1, 4, 512]))
            if self.load:
                self.text_features_tensor = torch.load('unnormal_seperate.pth').to(self._device)
                rest = torch.load('unnormal_seperate_2.pth').to(self._device)
                print(self.text_features_tensor.shape, rest.shape)
                self.text_features_tensor = torch.cat((self.text_features_tensor, rest), dim=0)
            self.label = self.label[:30]
            print(len(self.label))
            self.text_token =[]
            for l in self.label:
                a = []
                a.append(l)
                self.text_token.append(torch.cat([clip.tokenize(f"{c}") for c in a]))
        else:
            if self.seperate:
                print(len(self.label))
                self.text_token =[]
                for l in self.label:
                    a = []
                    a.append(l)
                    self.text_token.append(torch.cat([clip.tokenize(f"word {c}") for c in a]).to(self._device))
            else:
                self.label = random.sample(self.label, 3000)
                #self.label = self.label_origin
                #print(self.label)

                if self.padding:
                    self.label = self.label[:1]
                
                self.text_token = torch.cat([clip.tokenize(f"word {c}") for c in self.label])
    #@torch.jit.ignore
    #def no_weight_decay(self):
    #    param_names = {'text_embed.embedding.weight', 'pos_queries'}
    #    enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
    #    return param_names.union(enc_param_names)

    def clip_encode(self, img: torch.Tensor):
        #img = F.interpolate(img, size=224)
        #with torch.no_grad():
        emb = self.CLIPmodel.encode_image(img)
        return emb
        #return self.encoder(img)

    #def encode(self, img: torch.Tensor):
    #    return self.encoder(img)

    def txtencode(self, text: torch.Tensor):
        with torch.no_grad():
            emb = self.CLIPmodel.encode_text(text.to(self._device))
        return emb
    
    def decode(self, tgt: torch.Tensor, x: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None, GT: Optional[Tensor] = None):

        if GT is not None:
            tgt_list = GT
        else:
            if self.load_features:
                if self.new:
                    self.text_features = self.text_features.to(self._device)
                    self.text_features = self.text_features.to(torch.float32)
                    self.new = False

            elif self.seperate:
                if self.new:
                    text_features_list = []
                    for c in self.text_token:
                        text_feature = self.txtencode(c)
                        text_features_list.append(text_feature)
                    self.text_features = torch.cat(text_features_list, dim=0).to(self._device)
                    
                    #torch.save(self.text_features, 'real_seperate.pth')
                    
                    self.new = False
            elif self.text_pmt:
                if self.new:
                    #self.text_features_word = torch.cat([self.txtencode(c) for c in self.text_token]).to(self._device)
                    if self.load:
                        print("load unormal_seperate")
                    else:
                        text_features_list = []
                        for tokenized_text in self.text_token:
                            text_feature = self.txtencode(tokenized_text)
                            text_features_list.append(text_feature)
                            del text_feature

                        self.text_features_tensor = torch.cat(text_features_list, dim=0).to(self._device)
                        #torch.save(self.text_features_tensor, 'unnormal_seperate_4.pth')
                    self.new = False

                self.text_features_tensor = self.text_features_tensor.to(self._device)

                prompt = self.text_prompt
                prompt = prompt.expand(self.text_features_tensor.shape[0],-1,-1)
                #sos token 다음에 promt를 추가 sos+prompt+text
                text_features = torch.cat((self.text_features_tensor[:,:0,:], prompt, self.text_features_tensor[:,1:74,:]), dim=1)
                
                t = []
                for y in text_features:
                    #temp = y[torch.arange(y.shape[0]), self.text_token[i].argmax(dim=-1)] @ self.text_projection
                    #print(y.shape) #torch.Size([77, 512])
                    y = y.unsqueeze(0)
                    a = torch.arange(y.shape[0])
                    b = self.text_token[0].argmax(dim=-1)
                    c =  y[a,b]
                    temp = c @ self.text_projection
                    #temp =temp.squeeze()
                    t.append(temp)
                


                self.text_feature_unnormal = text_features
                self.text_features = torch.cat(t).to(self._device)
            else:
                if self.new:
                    self.text_features = self.txtencode(self.text_token)
                    self.tm = self.text_features[:][1]
                    self.text_features = self.text_features[:][0]
    
                    self.new = False

        tgt_list = []
        self.candidate_label = []
        for image_features in x:
            with torch.no_grad():
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True).to(self._device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            
            if self.contrastive:
                candidate_label = []
                _, candidate_idx = torch.topk(similarity, 4)
                self.candidate = self.text_features[candidate_idx]
                for i in candidate_idx:
                    candidate_label.append(self.text_token[i])
                self.candidate_label.append(candidate_label)
            _, indices = similarity.topk(1)
            tgt = self.text_feature_unnormal[indices]
            tgt_list.append(tgt)

        tgt_emb = torch.stack(tgt_list).to(self._device)

        #     tgt = self.label[indices]
        #     #tgt = self.tokenizer.encode(tgt, self._device)

        #     #Leehakho
        #     if self.padding:
        #         tgt = ""

        #     tgt_list.append(tgt)

        # tgt = self.tokenizer.encode(tgt_list, self._device)
        # tgt = tgt[:, :-1]
        # #tgt = torch.stack(tgt_list)

        B, N, L = tgt.shape
        #print(tgt_emb.shape)
        #N, L = tgt_emb.shape

        # # <bos> stands for the null context. We only supply position information for characters after <bos>.
        # null_ctx = self.text_embed(tgt[:, :1])

        # tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        #tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        tgt_emb = self.dropout(tgt_emb)


        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        #print(images.shape, max_length)
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        x, memory = self.clip_encode(images)
        #memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        #tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        # No prior context, so input is just <bos>. We query all positions.
        tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
        tgt_out = self.decode(tgt_in, x, memory,tgt_query=pos_queries)
        logits = self.head(tgt_out)
  
        return logits

    def write_unique_strings_to_file(self, strings):
        existing_strings = set()
        file_path = "/home/ohh/PycharmProject/PARCLIP/labels.txt"
        try:
            # 기존 파일에서 이미 있는 문자열 읽어오기
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
        #tgt_out = tgt[:, 1:]

        loss = 0
        loss_numel = 0
        n = (gt != self.pad_id).sum().item()
        max_len = tgt.shape[1] -2 # exclude <eos> from count

        out = self.forward(images, max_len)

        logits = out.flatten(end_dim=1)
        #print(logits.shape, gt.flatten().shape) #torch.Size([1664(고정), 37]) torch.Size([1024])
        loss = loss + (n * F.cross_entropy(logits, gt.flatten(), ignore_index=self.pad_id))
        loss_numel = loss_numel + n

        #tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
        #n = (tgt_out != self.pad_id).sum().item()

        #loss = F.cross_entropy(logits, gt.flatten(), ignore_index=self.pad_id)


        if self.contrastive:    
            from torch.cuda.amp import GradScaler, autocast
            torch.autograd.set_detect_anomaly(True)


            for i in labels[0]:
                if self.candidate_label[i]:
                    print("self.cand")

            x, con_labels = self.simclr.info_nce_loss(x, self.candidate)
            con_labels = con_labels.to(self._device)
            con_loss = self.criterion(x, con_labels)
            #scaler = GradScaler(enabled=True)
            #con_loss = scaler.scale(con_loss)
            #loss += 0.1* n * F.cross_entropy(x, con_labels, ignore_index=self.pad_id)


        loss /= loss_numel
        if self.contrastive:
            total_loss = 0.9 * loss + 0.1 * con_loss
        else:
            total_loss = loss

        self.log('loss', total_loss)
        #print(loss) #tensor(4.1242, device='cuda:6', grad_fn=<DivBackward0>)
        return total_loss
