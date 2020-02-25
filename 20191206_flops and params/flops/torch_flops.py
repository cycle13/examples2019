from torchvision.models import *
from thop import profile
import torch
 
model = vgg19()# resnet18() alexnet vgg16 resnet50
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))

print("params",params)
print("FLOPs",flops)
