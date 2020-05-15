# This repo contains codes for face recognition.

1. **models**: contains model archs; the resnets.py contains a modified version of resnet architecture, mainly change the first 7x7 kernel to smaller 3x3 kernel, and instead of using three conv in a block, this code use two conv in a block;

2. **utils**:
    * datautils.py: contains a custom dataset&dataloader api and some data augmentation functions;
    * layerutils.py: contains some custom layers;
    * lossutils.py: contains popular loss functions used in face recognition task, e.g. Arcface;

3. **solver.py**: training script with DistributedDataParallel support;



# TODO:
    * mix-precision training with pure pytorch, not using apex; (from pytorch 1.5, mix-precision is said to be supported)