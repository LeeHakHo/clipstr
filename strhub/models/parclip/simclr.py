import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

torch.manual_seed(2023)


class SimCLR(object):

    def __init__(self, device):
        self.args = 0#kwargs['args']
        self.batch_size = 256
        self.n_views = 2
        self.device = device
        self.temperature = 0.07

    def info_nce_loss(self, features, candidate):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        #Leehakho
        features = torch.permute(features, (1, 0))


        features = F.normalize(features, dim=1)
        #print(features.shape, labels.shape) #ttorch.Size([64, 512]) torch.Size([512, 512])

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        #print(mask.shape, similarity_matrix.shape)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
    
    def my_loss(self, features, candidate, label):
        #print(features.shape, candidate.shape, label.shape)
        #torch.Size([32, 512]) torch.Size([159, 512]) torch.Size([32, 512])

        batch = []
        for img, txt, lb in zip(features, candidate, label):
            #print(img.shape, txt.shape, lb.shape)
            txt = torch.cat((lb.unsqueeze(0), txt), dim=0)
            with torch.no_grad():
                txt /= txt.norm(dim=-1, keepdim=True)
                img /= img.norm(dim=-1, keepdim=True)
                similarity_matrix = torch.matmul(txt, img.T).to(self.device)
            similarity_matrix = similarity_matrix.unsqueeze(0)
            batch.append(similarity_matrix)  
        logits = torch.cat(batch,dim=0).to(self.device)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        #positive = torch.ones(1, dtype=torch.long).to(self.device)
        #labels = torch.cat((negative,positive),dim=0)
        #labels = labels.repeat(features.shape[0], 1) #batch,lens
        
        #print(logits.shape, labels.shape)

        # img = features
        # txt = torch.cat((candidate,label), dim=0)
        # with torch.no_grad():
        #     txt /= txt.norm(dim=-1, keepdim=True)
        #     img /= img.norm(dim=-1, keepdim=True)
        # #print(img.shape, txt.shape)
        # similarity_matrix = torch.matmul(txt, img.T)
        # #print(similarity_matrix)
        # batch.append(similarity_matrix)
        # logits = torch.cat(batch, dim = 0)
        # print(logits.shape) #torch.Size([191, 32])
        # negative = torch.zeros(logits.shape[0] -1, dtype=torch.long).to(self.device)
        # positive = torch.ones(1, dtype=torch.long).to(self.device)
        # labels = torch.cat((negative,positive),dim=0)
        # print(labels.shape) #torch.Size([192])
        return logits, labels