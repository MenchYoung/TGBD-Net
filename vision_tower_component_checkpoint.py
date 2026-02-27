from __future__ import annotations
from typing import Union
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from abc import ABC, abstractmethod
from torch import Tensor
import math
from typing import Any, Dict, List
import torch
import torch.nn as nn
from typing import Optional, Tuple, Type
from monai.networks.blocks import PatchEmbed
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets import ViT
from torchvision.models.video import swin3d_b
from torch.utils.checkpoint import checkpoint


def make_checkpointable(module):
    class CheckpointWrapper(nn.Module):
        def __init__(self, original_module):
            super().__init__()
            self.module = original_module 
            
        def forward(self, *args, **kwargs):
            if any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
                return checkpoint(self.module, *args, **kwargs, use_reentrant=False)
            else:
                return self.module(*args, **kwargs)
    
    return CheckpointWrapper(module)


class Swin3D(nn.Module):
    def __init__(self):
        super(Swin3D, self).__init__()
        self.model = swin3d_b()
        self.model.head = nn.Identity()
        self.model.avgpool = nn.Identity()
        

        self.feature_map_output = None # Cache the feature map at the input of the final norm layer (for downstream use).
        
        target_layer = self.model.norm
        
        target_layer.register_forward_hook(self.save_feature_map_hook)
        
    def save_feature_map_hook(self, module, input, output):
        """
        This function will be called when the norm layer runs
        It caches the 5D feature map (channel-last) for downstream modules
        """
        self.feature_map_output = input[0].permute(0, 4, 1, 2, 3) # -> [B, C, D, H, W]

    def forward(self, x):
        _ = self.model(x) 
        
        return self.feature_map_output
    
    def enable_gradient_checkpointing(self):
        print(f"⚡ [Swin3D] Analyzing model structure and injecting Gradient Checkpointing ...")
        print(f"   -> Number of Features Stages: {len(self.model.features)}")
        
        count = 0

        for i, layer in enumerate(self.model.features):

            if i <5:
                print(f"   -> skip {i}th layer")
                pass
            else:
                class_name = layer.__class__.__name__
                
                if "Sequential" in class_name:
                    is_transformer_stage = False
                    for submodule in layer:
                        if "SwinTransformerBlock" in submodule.__class__.__name__:
                            is_transformer_stage = True
                            break
                    
                    is_transformer_stage = True 

                    if is_transformer_stage:
                        print(f"   -> 🎯 {i}-th layer is Sequential, wrapping ...")
                        if not class_name.startswith("CheckpointWrapper"):
                            # Wrap a module with torch.utils.checkpoint to trade compute for memory during backprops
                            self.model.features[i] = make_checkpointable(layer)
                            count += 1
            
                
        if count > 0:
            print(f"⚡ [Swin3D] {count} Stages are wrapped successfully!")
        else:
            print("❌ [Swin3D] Error: No Stages are wrapped!")

    @property
    def hidden_size(self):
        return self.model.num_features
    
    @property
    def dtype(self):
        return self.model.features[0][0].weight.dtype

    @property
    def device(self):
        return self.model.features[0][0].weight.device
    



class Swin3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = Swin3D()

    def forward(self, images):
        last_feature = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        return image_features
    
    def enable_checkpointing(self):
        self.vision_tower.enable_gradient_checkpointing()

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size
    

from transformers import LlamaConfig


class LamedConfig(LlamaConfig):

    model_type = "llama"
