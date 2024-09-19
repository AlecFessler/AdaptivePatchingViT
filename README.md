With this particular setup, on cifar10 with no pretraining,
dps_vit_cifar10.py converges around 77% accuracy. The standard ViT
implementation (see std_vit_cifar10.py) converges around 74%
with the same hyperparameters and model size. The only difference
is the patch selection mechanism, and the positional embeddings
mechanism. This is because the dynamic patch selection mechanism
embeds the translation parameters for a patches affine transform
directly into the positional embeddings, because static positional
embeddings are not sufficient to capture the position. The standard
ViT was trained with learned positional embeddings added to the patches.
