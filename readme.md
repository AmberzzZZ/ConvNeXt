## official code startup

    official repo: facebook，2022，https://github.com/facebookresearch/ConvNeXt

    official lib version: 
        torch==1.8.0+cu111 
        torchvision==0.9.0+cu111 
        timm==0.3.2

    dataset: 分类数据集，按类别folder保存各类图片

    eval:
    python main.py --model convnext_base --eval true \
    --resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
    --input_size 224 --drop_path 0.2 \
    --data_path /path/to/imagenet-1k

    train:
    单机多卡/多机多卡：--nodes
    python -m torch.distributed.launch / run_with_submitit.py
    --batch_size: per GPU, 128 for convnext_base_224, 32 for convnext_base_384
    ema: use model_ema
    详细hyper setting在TRAINING.md

    train from scrach:
    train image-1k-224
    pretrain image-22k-224, fine-tune image-1k-384

    init:
    weight: trunc_normal(std=.02)
    bias: constant(0)

    model family:
    |    model   | resolution | acc on Image-1k | #params | FLOPs | drop |
    | ConvNeXt-T |     224    |       82.1      |   28M   |  4.5G |  0.1 |
    | ConvNeXt-S |     224    |       83.1      |   50M   |  8.7G |  0.4 |
    | ConvNeXt-B |     224    |       83.8      |   89M   | 15.4G |  0.5 |
    | ConvNeXt-B |     384    |       85.1      |   89M   | 45.0G |  0.5 |
    | ConvNeXt-L |     224    |       84.3      |  198M   | 34.4G |  0.5 |
    | ConvNeXt-L |     384    |       85.5      |  198M   |101.0G |  0.5 |
    














