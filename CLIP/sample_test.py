import os
import clip
import torch
from torchvision.datasets import CIFAR100

from torchvision.io.image import read_image
import torchvision
from torchvision import transforms
from PIL import Image
import string
import random
import numpy as np

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def images_to_tensor(image_paths, image_size=(32, 128)):
    """
    이미지들을 하나의 텐서로 변환하는 함수

    Parameters:
        image_paths (list): 이미지 파일 경로들이 담긴 리스트
        image_size (tuple): 이미지를 변환할 크기 (기본값: (224, 224))

    Returns:
        torch.Tensor: 이미지들이 쌓인 하나의 텐서
    """
    # 이미지 변환을 위한 torchvision의 transforms 객체 생성
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # 이미지들을 담을 빈 리스트 생성
    images = []

    # 이미지 파일 경로들을 순회하며 이미지들을 텐서로 변환하여 리스트에 추가
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).to(device)
        images.append(image_tensor)

    # 이미지 텐서들을 쌓아서 하나의 텐서로 만듦
    images_tensor = torch.stack(images)

    return images_tensor


# Download the dataset
#cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
#image, class_id = cifar100[3637]
#image_input = preprocess(image).unsqueeze(0).to(device)
#print(image_input.shape)
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
# print(cifar100.classes)

#img = read_image("/home/ohh/PycharmProject/CLIP-main/sample/tel.png")

label = ["black", "photo", "transform", "text", "shape", "tel", "锦素杜康", "chinese", "勇政的厨子", "岁潮童馆","afasdvbas","asgwqrf", "비즈니스class","비즈니스", "class",
          "해리the찬", "the", "해리", "korean", "english", "百年好舍tel", "百年好舍", "te百年好舍l", "tel百年好舍", "형사소송실무", "ocdl", "odlc", "clod","bafsvdaas","abasdf"]
# label = ["English", "Korean", "Chinese", "mixed language", "Japenese", "Both English and Korean", "single language"]
#label = ["English", "non-English"]
#label = string.digits + string.ascii_lowercase# + string.ascii_uppercase
#label = ["comfortable place "]

#img = Image.open("/home/ohh/PycharmProject/CLIP-main/sample/asgwqrf.png")
#image_input = preprocess(img).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"word {c}")for c in label]).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo containing the letter {c}")for c in label]).to(device)


#images_path = ["/home/ohh/PycharmProject/CLIP-main/sample/cafe.png", "/home/ohh/PycharmProject/CLIP-main/sample/cafe_a.png","/home/ohh/PycharmProject/CLIP-main/sample/cafe_b.png", "/home/ohh/PycharmProject/CLIP-main/sample/cafe_c.png"]
#images = images_to_tensor(images_path)

# Calculate features
with torch.no_grad():
    #image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    print(text_features[0])
    #sim = model(image=images, text=text_inputs)
sim = 0
scalar_list = []
for tensor in sim[0]:
    for value in tensor:
        scalar_list.append(value.item())
l = torch.tensor(scalar_list)

values, indices = torch.topk(l, k=0, dim=-1)

print("\nTop predictions:")
for value, index in zip(values, indices):
    print(f"{images_path[index]:>16s}: {value.item():.2f}%")


# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:")
for value, index in zip(values, indices):
    print(f"{label[index]:>16s}: {100 * value.item():.2f}%")


