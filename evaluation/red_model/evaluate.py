import ptflops
import torch
from model import REDModel

img_shape = (304, 240)
input_shape = (1, 10, *img_shape)  # event volume representation (batch_size, #bins, img width, img height)
model = REDModel(in_channels=10, img_shape=img_shape)

x = torch.rand(input_shape)
# _ = model.forward(x)

# Most of modern hardware architectures uses FMA instructions for operations with tensors.
# FMA computes a*x+b as one operation. Roughly GMACs = 0.5 * GFLOPs
# https://github.com/sovrasov/flops-counter.pytorch/issues/16#issuecomment-518585837
ptflops.get_model_complexity_info(model, input_shape[1:], verbose=True)
