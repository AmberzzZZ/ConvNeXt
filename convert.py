import torch
from convNext import convnext
import numpy as np


model_states = torch.load("weights/convnext_base_22k_224.pth", map_location='cpu')
print(model_states.keys())

model_weights = model_states['model']


# downsamp: stem + 3x stage downsamp, conv+ln
downsamp_weights = {k:v for k,v in model_weights.items() if 'downsamp' in k}

# stages: 4x stage blocks
stage_weights = {k:v for k,v in model_weights.items() if 'stages' in k}


# keras model
convnext_base = convnext(input_shape=(224,224,3), n_classes=21841,
                         depths=[3,3,27,3], widths=[128,256,512,1024], drop_block=0.5)
for layer in convnext_base.layers:
    if not layer.weights:
        continue

    # stem: conv + ln
    if 'stem' in layer.name:
        print(layer.name)
        if 'conv' in layer.name:
            torch_weights = [np.transpose(downsamp_weights['downsample_layers.0.0.weight'], (2,3,1,0)),
                             downsamp_weights['downsample_layers.0.0.bias']]
            layer.set_weights(torch_weights)
        else:
            torch_weights = [downsamp_weights['downsample_layers.0.1.weight'],
                             downsamp_weights['downsample_layers.0.1.bias']]
            layer.set_weights(torch_weights)
    elif 'downsamp' in layer.name:
        print(layer.name)
        stage_idx = layer.name.split('_')[-1]
        if 'conv' in layer.name:
            torch_weights = [np.transpose(downsamp_weights['downsample_layers.%s.1.weight' % stage_idx], (2,3,1,0)),
                             downsamp_weights['downsample_layers.%s.1.bias' % stage_idx]]
            layer.set_weights(torch_weights)
        else:
            torch_weights = [downsamp_weights['downsample_layers.%s.0.weight' % stage_idx],
                             downsamp_weights['downsample_layers.%s.0.bias' % stage_idx]]
            layer.set_weights(torch_weights)
    elif 'stage' in layer.name:
        stage_idx = layer.name.split('.')[0][-1]  # [0,1,2,3]
        block_idx = layer.name.split('.')[1][-1]  # start from 0
        print(layer.name, stage_idx, block_idx)

        block_weights = {k:v for k,v in stage_weights.items() if 'stages.%s.%s.' % (stage_idx,block_idx) in k}
        torch_weights = [[np.transpose(stage_weights['stages.%s.%s.dwconv.weight' % (stage_idx,block_idx)], (2,3,0,1)),   # dwconv
                         stage_weights['stages.%s.%s.dwconv.bias' % (stage_idx,block_idx)]],
                         [stage_weights['stages.%s.%s.norm.weight' % (stage_idx,block_idx)],   # ln
                         stage_weights['stages.%s.%s.norm.bias' % (stage_idx,block_idx)]],
                         [np.transpose(stage_weights['stages.%s.%s.pwconv1.weight' % (stage_idx,block_idx)], (1,0)).unsqueeze(0).unsqueeze(0),   # conv1
                         stage_weights['stages.%s.%s.pwconv1.bias' % (stage_idx,block_idx)]],
                         [np.transpose(stage_weights['stages.%s.%s.pwconv2.weight' % (stage_idx,block_idx)], (1,0)).unsqueeze(0).unsqueeze(0),   # conv2
                         stage_weights['stages.%s.%s.pwconv2.bias' % (stage_idx,block_idx)]],
                         [stage_weights['stages.%s.%s.gamma' % (stage_idx,block_idx)]],    # gamma
        ]
        # print(len(torch_weights), [len(i) for i in torch_weights])
        cnt = 0
        for sub_l in layer.layers:
            if not sub_l.weights:
                continue
            print('\t', sub_l.name, cnt)
            sub_l.set_weights(torch_weights[cnt])
            cnt += 1
    elif 'final_norm' in layer.name:
        torch_weights = [model_weights['norm.weight'], model_weights['norm.bias']]
        layer.set_weights(torch_weights)
    elif 'head' in layer.name:
        torch_weights = [np.transpose(model_weights['head.weight'], (1,0)),
                         model_weights['head.bias']]
        layer.set_weights(torch_weights)
convnext_base.save_weights("weights/convnext_base_22k_224.h5")














