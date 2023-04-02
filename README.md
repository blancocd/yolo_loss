# Yolo
We rely on a pretrained classifier as the backbone for our detection network. PyTorch offers a variety of models which are pretrained on ImageNet in the [`torchvision.models`](https://pytorch.org/docs/stable/torchvision/models.html) package. In particular, we will use the ResNet50 architecture as a base for our detector. This is different from the base architecture in the Yolo paper and also results in a different output grid size (14x14 instead of 7x7).

Models are typically pretrained on ImageNet since the dataset is very large (> 1 million images) and widely used. The pretrained model provides a very useful weight initialization for our detector, so that the network is able to learn quickly and effectively.

The training dataset is PASCAL's training and validation set as it is pretty small to only use the training dataset. Be sure to run the download_data.sh script! See the noobj0.1.txt file to check what the output should look like across 50 epochs

Finally, if you only want to use the trained model with your own images, use the visualize.ipynb notebook. No GPU required.

Credit to CS444 staff for started code.
