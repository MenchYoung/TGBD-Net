import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import os

try:
    from vision_tower_component_checkpoint import LamedConfig
    from vision_tower_component_checkpoint import Swin3DTower
except ImportError:
    pass

class PrototypeClustering(nn.Module):
    def __init__(self, feature_dim, num_prototypes=4):
        super().__init__()
        self.num_prototypes = num_prototypes
        # Define K bridging anchors
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, feature_dim))
        nn.init.orthogonal_(self.prototypes)
        

    def forward(self, z_attn):
        # 1. Normalize
        z_norm = F.normalize(z_attn, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1) # (K, C)

        similarity = torch.mm(z_norm, proto_norm.t()) # (B, K)
        
        return similarity
    

class TGBDModel(nn.Module):
    def __init__(self, vision_tower_config, weights_path, num_classes=2, dropout_rate=0.3):
        """
        - vision_tower_config: Swin3D Model config
        - weights_path: default weights path
        - num_classes: number of classes (e.g., EGFR mutant/wildtype -> 2)。
        """
        super().__init__()
        
        # --- Load and Freeze Backbone ---
        # Freeze backbone (can be unfrozen later via unfreeze_encoder)
        self.vision_encoder = Swin3DTower(vision_tower_config)
        
        print(f"  -> Loading weights from '{weights_path}' for Backbone...")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.vision_encoder.load_state_dict(new_state_dict, strict=False)
        print("  -> Weights loaded for Backbone Successfully.")

        self.vision_encoder.enable_checkpointing()
        
        # Freeze Backbone in the early stage
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("  -> ⚠️ Backbone is frozen, now training the learners...")
        
        self.d_model = self.vision_encoder.hidden_size
        
        
        # --- Bidirectional Decoupled Module ---
        # 1. learnable tumor query token
        self.tumor_query_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.tumor_query_token, std=0.02)

        #cluster head
        self.cluster_head = PrototypeClustering(self.d_model)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model), 
            nn.LayerNorm(self.d_model),                
            nn.ReLU(),                                
            nn.Dropout(dropout_rate)                   
        )

        # 2. Cross-Attention for Tumor-Guided Feature Extraction Module (TGF)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=8, 
            dropout=0.1, 
            batch_first=True
        )

        # 3. Auxiliary head for generating attention maps (optional, but recommended)
        self.auxiliary_head = nn.Sequential(
            nn.Conv3d(self.d_model, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.d_model // 2, num_classes)
        )
        self.cox_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.d_model // 2, 1)
        )

        print(f"  -> EGFR classifier initialized, output classes: {num_classes}")
        print("\n--- TGBD Model Initialized ---\n")

    def forward(self, image_global, image_local=None):

        # =================================================
        # Path A：Global Path
        # =================================================
        # 1. Extract Global Feature Map
        feat_map_global = self.vision_encoder(image_global) 

        # 2. Obtain Z_global (f_g)
        z_global_ctx = F.adaptive_avg_pool3d(feat_map_global, (1, 1, 1)).flatten(1)
       
        
        # 3. Obtain Z_attn (f_tg)
        # Query = Global Context + Learnable Token
        query = z_global_ctx.unsqueeze(1) + self.tumor_query_token
        
        # Key/Value = Flattened Global Feature Map
        seq_global = feat_map_global.flatten(2).permute(0, 2, 1)
        
        # Cross Attention
        z_attn, _ = self.cross_attention(query = query, key = seq_global,value = seq_global)
        z_attn = z_attn.squeeze(1) # (B, 1024) 

        # =================================================
        # Path B：Local Path - BBox
        # =================================================
        z_crop = None
        if image_local is not None:
            # 1. Extract Local Feature Map with Shared Backbone 
            feat_map_local = self.vision_encoder(image_local)
            
            # 2. Obtain Z_crop (f_crop)
            z_crop = F.adaptive_avg_pool3d(feat_map_local, (1, 1, 1)).flatten(1) # (B, 1024)

        # =================================================
        # Fusion and Projection
        # =================================================
        # Obtain Z_final (f_shared)
        z_final = torch.cat([z_global_ctx, z_attn], dim=1) # (B, 2048)
        z_final = self.projector(z_final) # (B, 1024)

        proto_sim = None
        if self.training:
            # calculate prototype similarity
            proto_sim = self.cluster_head(z_final)
        
        # 分类/预后
        outputs = {
            "logits": self.classifier(z_final),  
            "pred_risk": self.cox_head(z_final) if hasattr(self, 'cox_head') else None,
            "z_attn": z_attn,  # for calculating alagnment loss
            "z_crop": z_crop,   # for calculating alagnment loss
            #"attn_map": attn_map 
            "proto_sim":proto_sim
        }
        
        return outputs


    def unfreeze_encoder(self, num_layers_to_unfreeze=2):
        """
        function to unfreeze backbone,
        num_layers_to_unfreeze: number of layers to unfreeze
        """
        print(f"\n🔓 unfeezing Backbone's last {num_layers_to_unfreeze} layers...")
        
        try:
            encoder_stages = self.vision_encoder.vision_tower.model.features
            num_total_stages = len(encoder_stages)
            
            start_idx = max(0, num_total_stages - num_layers_to_unfreeze)
            
            for i in range(start_idx, num_total_stages):
                for param in encoder_stages[i].parameters():
                    param.requires_grad = True
            
            print(f"  -> unfreezing Stage from {start_idx} to {num_total_stages-1} successfully.")
            
        except AttributeError:
            print("  -> ⚠️ Warning: could not find 'vision_tower.model.features' in backbone. Unfreezing all layers...")
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

