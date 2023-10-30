import torch
from torch import cat as cat

torch.manual_seed(2023)


class SimCLR(object):

    def __init__(self, device):
        self.device = device
        self.temperature = 0.07

    @torch.jit.script
    def dotProduct(image, text):
        return (image @ text.T)

    def my_loss(self, features, candidate, label):
        batch = []
        for img, txt, lb in zip(features, candidate, label):
            txt = torch.cat((lb.unsqueeze(0), txt), dim=0)
            with torch.no_grad():
                txt /= txt.norm(dim=-1, keepdim=True)
                img /= img.norm(dim=-1, keepdim=True)
                similarity_matrix = (img @ txt.T).to(self.device)
            similarity_matrix = similarity_matrix.unsqueeze(0)
            batch.append(similarity_matrix)  
        logits = torch.cat(batch,dim=0).to(self.device)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        return logits, labels