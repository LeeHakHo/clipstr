from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

model = load_from_checkpoint("/home/ohh/PycharmProject/parseq-main/outputs/parseq/par eng_cn, eng_cn 300/checkpoints/epoch=251-step=145578-val_accuracy=72.4851-val_NED=84.4957.ckpt", **kwargs).eval().to('cuda')
# Get your input
img = read_image("/home/ohh/PycharmProject/parseq-main/samples/00009_0000.png")
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
target_layers = [model.layer4[-1]]

cam_extractor =SmoothGradCAMpp(model, target_layers)

# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()