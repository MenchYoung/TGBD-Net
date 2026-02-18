import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
import itertools
import torch.distributed as dist
from lifelines.utils import concordance_index
from scipy.optimize import linear_sum_assignment
import sys

def cox_loss(risk_score, times, events):
    """
    Cox Loss:
    1. # Skip if sample size < 2 (partial likelihood undefined / unstable)
    2. use view(-1) instead of squeeze(), to avoid 0-dim tensor
    """
    # --- 1. filter invalid data ---
    mask = times > 0
    num_valid = mask.sum()

    # At least 2 samples are required to compute Cox partial likelihood
    # With only 1 sample, risk_score[mask] leads to dimension collapse and partial likelihood cannot be computed
    if num_valid < 2:
        return risk_score.sum() * 0.0 
    
    risk_s = risk_score[mask].view(-1)
    time_s = times[mask].view(-1)
    event_s = events[mask].view(-1)
    
    # --- 2. Check if there is any event ---
    if event_s.sum() == 0:
        return risk_score.sum() * 0.0

    # --- 3. Sort by time ---
    sorted_idx = torch.argsort(time_s, descending=True)
    risk_sorted = risk_s[sorted_idx]
    event_sorted = event_s[sorted_idx]
    
    # --- 4. Compute Log-Sum-Exp ---
    exp_risk = torch.exp(risk_sorted)
    risk_set_sum = torch.cumsum(exp_risk, dim=0)
    log_risk_set = torch.log(risk_set_sum + 1e-8) 
    
    # --- 5. Compute NLL ---
    num_events = event_sorted.sum()
    nll = -torch.sum(event_sorted * (risk_sorted - log_risk_set)) / num_events
    
    return nll

class EGFRTrainer:
    def __init__(self, model, dataloaders, config, accelerator,num_of_clusters):
        self.accelerator = accelerator
        self.config = config
        self.model = model 
        self.dataloaders = dataloaders
        
        
        # Loss
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_align = nn.CosineEmbeddingLoss(margin=0.0, reduction='none')

        # Soft Lable
        self.num_prototypes = num_of_clusters 
        # 1. EGFR-related attribute (mutation rate for bridging anchors) initialized as 0.5 
        self.proto_egfr = torch.full((self.num_prototypes,), 0.5).to(accelerator.device)
        # 2. PFS-related attribute (average risk for bridging anchors) initialized as 0.0
        self.proto_risk = torch.zeros(self.num_prototypes).to(accelerator.device)
        # 3. momentum (0.9)
        self.momentum = 0.9
        self.cluster_weight = 0.0


        self.alpha = getattr(config, 'ALPHA_ALIGN', 1.0)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LR, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=(config.EPOCHS - config.WARMUP_EPOCHS)*2,eta_min=1e-6
        )
        print(f"DEBUG: Scheduler T_max = {self.scheduler.T_max}")

        # --- Prepare DDP ---
        self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['val'], self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloaders['train'], self.dataloaders['val'], self.scheduler
        )
        if 'test' in self.dataloaders:
            self.dataloaders['test'] = self.accelerator.prepare(self.dataloaders['test'])

        if self.accelerator.is_main_process:
            print(f"--- EGFR Trainer Ready (DDP Mode) | Align Alpha: {self.alpha} ---")

    
    
    def perform_kmeans_initialization(self):
        from DataSet.DataSet_FULL import get_kmeans_dataloaders
        
        clean_loader = get_kmeans_dataloaders(
            csv_path=self.config.CSV_PATH,   
            image_size=self.config.IMAGE_SIZE, 
            batch_size=self.config.BATCH_SIZE, 
            num_workers=4
        )
        clean_loader = self.accelerator.prepare(clean_loader)
        log_file = "kmeans_initialization_log.txt"
    
        # Auxiliary function: print and save to file
        def log_print(message, mode='a'):
            print(message)
            with open(log_file, mode, encoding='utf-8') as f:
                f.write(message + '\n')

        if self.accelerator.is_main_process:
            print(f"\n⚡ [Epoch {self.config.WARMUP_EPOCHS}] Backbone Warming Up Complete. Starting K-means Initialization...")
            
        self.model.eval()
        local_feats = []
        
        # Feature Extraction 
        with torch.no_grad():
            for batch in tqdm(clean_loader, desc="Extracting Features", disable=not self.accelerator.is_main_process):
                img_global = batch['image_global']
                img_local = batch['image_local']
                
                outputs = self.model(image_global=img_global, image_local=img_local)
                z_attn = outputs['z_attn']
                
                local_feats.append(z_attn)
        
        # Concatenate all local features
        if len(local_feats) > 0:
            local_feats = torch.cat(local_feats, dim=0) # (N_local, 1024)
        else:
            local_feats = torch.tensor([]).to(self.accelerator.device)

        # DDP Gather：gather all local features
        # This aggregates features across all processes to form a global feature set
        all_feats = self.accelerator.gather(local_feats)

        best_centers = torch.zeros(
            (self.num_prototypes, all_feats.shape[-1]), 
            device=self.accelerator.device
        )

        # Run K-Means
        if self.accelerator.is_main_process:
            print("🤖 Main Process : Performing K-Means...")
            
            # ---  Center features by subtracting the global mean ---
            feats_tensor = all_feats
            mean_feat = feats_tensor.mean(dim=0, keepdim=True)
            centered_feats = feats_tensor - mean_feat

            std_center_val = centered_feats.std(dim=0).mean().item()
            log_print(f"📊 std after Subtract mean: {std_center_val:.6f}")
            sim_center_matrix = torch.mm(centered_feats[:10], centered_feats[:10].t())
            log_print(f"👀 The similarity matrix of the top 10 samples after Subtract mean:\n{sim_center_matrix}")
            
            # transfer to numpy
            feats_np = centered_feats.cpu().numpy()
            
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.num_prototypes, n_init=20, random_state=42)
            kmeans.fit(feats_np)
            
            # obtain cluster centers 
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.accelerator.device)
            
            # ---  Add mean ---
            real_centers = centers + mean_feat
            real_centers = F.normalize(real_centers, dim=1) # normalize
            
            best_centers.copy_(real_centers)
            print(f"✅ K-Means Initialization Complete. Broadcasting...")

        dist.broadcast(best_centers, src=0)
        
        # 1. Obtain Old Prototypes
        if hasattr(self.model, 'module'):
            head = self.model.module.cluster_head
        else:
            head = self.model.cluster_head
            
        old_protos = head.prototypes.data.clone() 
            
        # 2. Matching Only on Main Process
        indices_map = torch.arange(self.num_prototypes).to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            # Compute Cost Matrix
            old_norm = F.normalize(old_protos, dim=1)
            new_norm = F.normalize(best_centers, dim=1)
            sim_matrix = torch.mm(old_norm, new_norm.t())
            cost_matrix = 1.0 - sim_matrix.cpu().numpy()
            
            # Hungarian Matching
            from scipy.optimize import linear_sum_assignment 
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            indices_map = torch.tensor(col_ind).to(self.accelerator.device)
            print(f"🔄 index for resort bridging anchors: {indices_map.tolist()}")

        # 3. Broadcast
        dist.broadcast(indices_map, src=0)
        
        # 4. Resort best_centers
        sorted_centers = best_centers[indices_map]
        

        # 5. Soft Update
        # Define Alpha
        alpha = 0.9
        
        # Obtain New Bridging Anchors
        # New = Old * 0.9 + Calculated * 0.1
        new_protos = alpha * head.prototypes.data + (1 - alpha) * sorted_centers
        
        # Normalize
        new_protos = F.normalize(new_protos, dim=1)
        
        head.prototypes.data.copy_(new_protos)
        
        # Freeze Bridging Anchors
        head.prototypes.requires_grad = False
        
        if self.accelerator.is_main_process:
            print(f"✅ Soft Update Complete. (Alpha={alpha})")

            
        # 6. Print Check Values (Optional)
        rank = self.accelerator.process_index
        check_values = head.prototypes.data[0, :5].cpu().numpy()

        log_print(f"🆔 [GPU {rank}]: top 5 numbers of Bridging Anchors: {check_values}")
        sys.stdout.flush() 

        self.accelerator.wait_for_everyone()
        
        del clean_loader
        del all_feats
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            print(f"✅ K-Means Initialization Complete! Please check your Bridging Anchors.")
        
    
    def _ddp_update_memory(self, mask, values, memory_bank, proto_sim):
        """
        Auxiliary function to update the EGFR-related and PFS-related attributes
        """
        momentum = self.momentum 
        device = self.accelerator.device
        K = self.num_prototypes
        
        # 1. Initialize local statistics
        local_sum = torch.zeros(K, device=device)
        local_count = torch.zeros(K, device=device)

        # Compute local statistics
        if mask.sum() > 0:
            sim_valid = proto_sim[mask]                 
            val_valid = values[mask].view(-1)           
            assigns = torch.argmax(sim_valid, dim=1)    

            for k in range(K):
                k_mask = (assigns == k)
                if k_mask.sum() > 0:
                    local_sum[k] = val_valid[k_mask].sum()
                    local_count[k] = k_mask.sum()

        # 2. DDP synchronization
        if self.accelerator.num_processes > 1:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        # 3. Update 
        update_mask = (local_count > 0)
        
        if update_mask.sum() > 0:

            idx = torch.where(update_mask)[0]
            
            # Compute average in the current batch
            # current_avg has the same length as idx
            current_avg = local_sum[idx] / (local_count[idx] + 1e-8)
            
            memory_bank[idx] = momentum * memory_bank[idx] + (1 - momentum) * current_avg
    
    def _train_epoch(self, epoch):
        """ Training Loop for One Epoch """
        self.model.train()
        total_loss = 0
        
        # Warmup
        if epoch < self.config.WARMUP_EPOCHS:
            warmup_lr = self.config.LR * (epoch + 1) / self.config.WARMUP_EPOCHS
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Unfreeze
        if epoch == self.config.UNFREEZE_EPOCH:
            if self.accelerator.is_main_process:
                print(f"\n🔓 [Epoch {epoch}] Unfreezing Backbone...")
            raw_model = self.accelerator.unwrap_model(self.model)
            raw_model.unfreeze_encoder(num_layers_to_unfreeze=2)


        pbar = tqdm(self.dataloaders['train'], desc=f"Ep {epoch}/{self.config.EPOCHS} [Train]", disable=not self.accelerator.is_main_process)
        
        for batch in pbar:
            img_global = batch['image_global']
            img_local  = batch['image_local']
            has_bbox   = batch['has_bbox']
            labels     = batch['label_egfr']
            pfs_time   = batch['pfs_time']
            pfs_event  = batch['pfs_event']
            
            self.optimizer.zero_grad()
            outputs = self.model(image_global=img_global, image_local=img_local)
            
        # Cls loss
            # Compute Cls loss only when label is not -1
            mask_cls = (labels != -1)
            
            # Compute Cls loss only when there is at least one valid sample, otherwise cls loss is 0
            if mask_cls.sum() > 0:
                loss_cls = self.criterion_cls(outputs['logits'][mask_cls], labels[mask_cls])
            else:
                loss_cls = torch.tensor(0.0).to(self.accelerator.device)
            

        # Align loss 
            z_attn = outputs['z_attn']
            z_crop = outputs['z_crop']

            target_ones = torch.ones(img_global.size(0)).to(self.accelerator.device)
            loss_align_raw = self.criterion_align(z_attn, z_crop, target_ones)
 
            mask = has_bbox.float()
            valid_samples = mask.sum()
            loss_align = (loss_align_raw * mask).sum() / valid_samples if valid_samples > 0 else torch.tensor(0.0).to(self.accelerator.device)
            
        # Cox loss
            pred_risk = outputs['pred_risk']
            loss_cox = cox_loss(pred_risk,pfs_time,pfs_event)
            

        # Distillation loss
            proto_sim = outputs['proto_sim'] # (B, K)

            # Mask
            mask_has_egfr = (labels != -1)
            mask_has_pfs  = (pfs_time > 0)
            mask_no_egfr  = (labels == -1)
            mask_no_pfs   = (pfs_time <= 0)

            # Initialize Loss 
            loss_distill_egfr = torch.tensor(0.0).to(self.accelerator.device)
            loss_distill_pfs = torch.tensor(0.0).to(self.accelerator.device)

            if self.cluster_weight > 0:

                # ---------------------------------------------------
                # Stage (1) Joint-Attribute Coupling: update anchor attributes (EMA) using labeled samples
                # ---------------------------------------------------
                with torch.no_grad():

                    # >>> Update EGFR-related attribute
                    self._ddp_update_memory(
                        mask=mask_has_egfr, 
                        values=labels.float(), 
                        memory_bank=self.proto_egfr,
                        proto_sim = proto_sim

                    )

                    # >>> Update PFS-related attribute
                    self._ddp_update_memory(
                        mask=mask_has_pfs, 
                        values=pred_risk, 
                        memory_bank=self.proto_risk,
                        proto_sim = proto_sim
                    )

                # ---------------------------------------------------
                # Stage (2) Soft Label Assignment: compute soft pseudo-labels for unlabeled samples and distillation loss
                # ---------------------------------------------------
                # Softmax
                soft_weights = F.softmax(proto_sim / 0.2, dim=1)

                # --- EGFR-related distillation ---
                # Distill only when there is at least one valid sample
                if mask_no_egfr.sum() > 0:

                    # 1. Compute target
                    target = torch.matmul(soft_weights[mask_no_egfr], self.proto_egfr.detach())
                    
                    # 2. Compute predicted EGFR probability (logit -> sigmoid)
                    pred = torch.sigmoid(outputs['logits'][mask_no_egfr][:, 1])
                    
                    # 3. MSE
                    loss_distill_egfr = F.mse_loss(pred, target)

                # --- PFS-related distillation ---
                if mask_no_pfs.sum() > 0:
                    # 1. Compute target
                    target = torch.matmul(soft_weights[mask_no_pfs], self.proto_risk.detach())
                    
                    # 2. Compute risk score
                    pred = pred_risk[mask_no_pfs].view(-1)
                    
                    # 3. MSE
                    loss_distill_pfs = F.mse_loss(pred, target)

            # ---------------------------------------------------


            loss = loss_cls  + loss_cox + self.cluster_weight * (loss_distill_pfs+loss_distill_egfr) + self.alpha * loss_align
            
            self.accelerator.backward(loss)


            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()


            total_loss += loss.item()

        # print loss
            l_cls = loss_cls.item()
            l_cox = loss_cox.item()
            l_ali = loss_align.item()
            l_dis = (loss_distill_egfr+loss_distill_pfs).item()
            
            pbar.set_postfix({
                'Tot': f'{loss.item():.2f}',
                'Cls': f'{l_cls:.3f}', 
                'Cox': f'{l_cox:.3f}',
                'Ali': f'{l_ali:.3f}',
                'Dis': f'{l_dis:.3f}'
            })
            
        if self.scheduler and epoch >= self.config.WARMUP_EPOCHS:
            self.scheduler.step()
            
        return total_loss / len(self.dataloaders['train'])

    def _evaluate_epoch(self, mode='val'):
        self.model.eval()
        if mode not in self.dataloaders:
            return 0.0, 0.0, 0.0, None
            
        loader = self.dataloaders[mode]
        
        # 1. Initialize local variables 
        local_ids, local_labels, local_probs = [], [], []
        local_risks, local_times, local_events = [], [], []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"   -> Eval [{mode}]", disable=not self.accelerator.is_main_process)
            for batch in pbar:
                img_global = batch['image_global']
                img_local  = batch['image_local']
                labels     = batch['label_egfr']
                ids        = batch['id']
                pfs_time   = batch['pfs_time']
                pfs_event  = batch['pfs_event']
                
                outputs = self.model(image_global=img_global, image_local=img_local)
                
                probs = F.softmax(outputs['logits'], dim=1)[:, 1]
                risks = outputs.get('risk_score', outputs.get('pred_risk'))
                
                local_probs.append(probs)
                local_labels.append(labels)
                local_ids.extend(ids)
                local_risks.append(risks)
                local_times.append(pfs_time)
                local_events.append(pfs_event)

        # 2. DDP Gather
        def gather_tensor(tensor_list):
            if len(tensor_list) > 0:
                t = torch.cat(tensor_list)
            else:
                t = torch.tensor([]).to(self.accelerator.device)
            return self.accelerator.gather(t)

        all_probs = gather_tensor(local_probs)
        all_labels = gather_tensor(local_labels)
        all_risks = gather_tensor(local_risks)
        all_times = gather_tensor(local_times)
        all_events = gather_tensor(local_events)
        
        # Gather ID 
        all_ids = []
        if self.accelerator.num_processes > 1:
            try:
                if hasattr(self.accelerator, 'gather_object'):
                    all_ids_nested = self.accelerator.gather_object(local_ids)
                    all_ids = list(itertools.chain.from_iterable(all_ids_nested))
                else:
                    output_ids_list = [None for _ in range(self.accelerator.num_processes)]
                    dist.all_gather_object(output_ids_list, local_ids)
                    all_ids = list(itertools.chain.from_iterable(output_ids_list))
            except Exception:
                if self.accelerator.is_main_process:
                    all_ids = local_ids
        else:
            all_ids = local_ids

        # 3. Convert to Numpy
        all_probs_np = all_probs.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()
        all_risks_np = all_risks.cpu().numpy().flatten()
        all_times_np = all_times.cpu().numpy().flatten()
        all_events_np = all_events.cpu().numpy().flatten()

        # 4. Compute Metrics and deduplication
        auc, acc, c_index = 0.5, 0.0, 0.5
        predictions_df = None

        if self.accelerator.is_main_process:
            # A. Construct DataFrame
            min_len = min(len(all_ids), len(all_labels_np))
            df = pd.DataFrame({
                'id': all_ids[:min_len],
                'true_label': all_labels_np[:min_len],
                'pred_prob': all_probs_np[:min_len],
                'pred_risk': all_risks_np[:min_len],
                'pfs_time': all_times_np[:min_len],
                'pfs_event': all_events_np[:min_len]
            })
            
            # B. Deduplication with id
            # reserve the first occurrence
            df_clean = df.drop_duplicates(subset=['id']).reset_index(drop=True)
            
            # C. Save DataFrame 
            predictions_df = df_clean

            # D. Compute AUC using clean data
            cls_df = df_clean[df_clean['true_label'] != -1]
            if len(cls_df) >= 2 and len(cls_df['true_label'].unique()) > 1:
                try:
                    auc = roc_auc_score(cls_df['true_label'], cls_df['pred_prob'])
                    preds = (cls_df['pred_prob'] > 0.5).astype(int)
                    acc = accuracy_score(cls_df['true_label'], preds)
                except:
                    pass

            # E. Compute C-index using clean data
            cox_df = df_clean[df_clean['pfs_time'] > 0]
            if len(cox_df) > 2:
                try:
                    c_index = concordance_index(
                        cox_df['pfs_time'],
                        -cox_df['pred_risk'], 
                        cox_df['pfs_event']
                    )
                except ImportError:
                    print("⚠️ Install lifelines for C-index")
                except Exception:
                    pass
        
        return auc, acc, c_index, predictions_df

    def fit(self):
        if self.accelerator.is_main_process:
            os.makedirs(self.config.SAVE_DIR, exist_ok=True)
            print(f"\n🚀 Start Training (DDP Mode)...")
            print(f"📂 Save Directories: {self.config.SAVE_DIR}")
        
        history = []
        best_val_auc = -1.0
        epochs_no_improve = 0
        
        # Save Directories
        path_log = os.path.join(self.config.SAVE_DIR, 'training_log.csv')
        path_best_model = os.path.join(self.config.SAVE_DIR, 'best_model.pth')
        path_best_val_csv = os.path.join(self.config.SAVE_DIR, 'best_val_preds.csv')
        path_best_test_csv = os.path.join(self.config.SAVE_DIR, 'best_test_preds.csv')

        for epoch in range(self.config.EPOCHS):

            if epoch >= 10 and (epoch - 10) % 5 == 0 and epoch <= 100:
                self.perform_kmeans_initialization()
                self.optimizer.state.clear()

            if epoch > self.config.WARMUP_EPOCHS -1:
                self.cluster_weight = 1.0

            


            # 1. Train
            train_loss = self._train_epoch(epoch)
            
            # 2. Val
            val_auc, val_acc, val_cindex,val_df = self._evaluate_epoch('val')
            
            # 3. Test 
            test_auc, test_acc, test_cindex,test_df = self._evaluate_epoch('test')
            
            # 4. DDP Sync
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print("-" * 80)
                print(f"Epoch {epoch:03d} | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f}")
                print(f"   >> Val  AUC: {val_auc:.4f} | Val  C-idx: {val_cindex:.4f}")
                print(f"   >> Test AUC: {test_auc:.4f} | Test C-idx: {test_cindex:.4f}")
                print("-" * 80)
                
                history.append({
                    'epoch': epoch,
                    'lr': current_lr,
                    'train_loss': train_loss,
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'val_cindex':val_cindex,
                    'test_auc': test_auc, 
                    'test_acc': test_acc,
                    'test_cindex':test_cindex 
                })
                pd.DataFrame(history).to_csv(path_log, index=False)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    epochs_no_improve = 0
                    
                    print(f"  🔥 Best Val AUC! Saving model and predictions...")
                    
                    # 1. Save Model
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    torch.save(unwrapped_model.state_dict(), path_best_model)
                    
                    # 2. Save predictions
                    if val_df is not None:
                        val_df.to_csv(path_best_val_csv, index=False)
                        
                    if test_df is not None:
                        test_df.to_csv(path_best_test_csv, index=False)
                else:
                    epochs_no_improve += 1
                
                # Early Stopping
                if epochs_no_improve >= self.config.EARLY_STOP_PATIENCE:
                    print(f"🛑 Early Stopping at Epoch {epoch}")
                    break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(fr"\n🎉 Training Completed! Check {self.config.SAVE_DIR} for best_val_preds.csv and best_test_preds.csv")