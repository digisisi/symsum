This repository accompanies the paper **Say Goodbye to Gradient Vanishing**.  
You can use it to train ResNet50 and ResNet101 models on either CIFAR10, CIFAR100, or the Tiny ImageNet dataset.  
All the configuration options are passed through the command line; (there is no configuration file).  
Please refer to the help field of the arguments to understand what options are available and what they do.

All the artifacts will be logged in `./artifacts` directory.
Metrics are logged in Weights & Biases.  
Datasets should go in `./data`. CIFAR datasets are automatically downloaded there.
The Tiny ImageNet dataset should be manually put there.  
Implementation of SymSum and its variants can be found in `models/activations.py`.

The code is tested with Python 3.8, PyTorch 1.8.1 and Torchvision 0.9.1 with CUDA 11.2.  
Minor modifications are required to run with the latest PyTorch because some modules have been moved.

Example command:
```
python train.py -batch_size 64 -epochs 120 -model res50 -norm_layer batch -bn_affine -bn_rs -lr .05 -weight_decay 5e-5 -augs mir randcrop colorjit -activation symsum -cifar 100 -symmetry_corr 1 -init_gain_param 1.4
```

