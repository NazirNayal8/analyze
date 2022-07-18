import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from typing import Callable, Optional
from torchmetrics import JaccardIndex
from tqdm import tqdm

import wandb
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from ood_metrics import fpr_at_95_tpr

class MaxLogitsAnalyzer:

    def __init__(
        self, 
        model: nn.Module, 
        ckpt_path: str, 
        model_ckpt_loader: Callable,
        inference_func: Callable,
        config: Optional[dict]
    ):
        
        self.model = model_ckpt_loader(model, ckpt_path, config)
        self.config = config
        self.inference_func = inference_func
        
    
    def get_logits(self, x, **kwargs):
        return self.inference_func(self.model, x, **kwargs)


    def evaluate_mIoU(
        self,
        dataset,
        num_classes, 
        ignore_index, 
        batch_size=1,
        device=torch.device('cpu'),
        num_workers=1,
        **kwargs
    ):

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        mIoU_metric = JaccardIndex(num_classes=num_classes, ignore_index=ignore_index).to(device)
        
        for x, y in tqdm(loader):
        
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():

                logits = self.get_logits(x, **kwargs)
            
            dummy_extension = torch.zeros(logits.shape[0], 1, logits.shape[2], logits.shape[3]).to(device)
            logits = torch.cat([logits, dummy_extension], dim=1)

            mIoU_metric.update(logits, y)
            
        return mIoU_metric.cpu().compute()

    def extract_max_logits_ood(self, loader, device=torch.device('cpu'), skip=1, **kwargs):

        max_logits_id = []
        max_logits_ood = []
    
        for x, y in tqdm(loader, desc="Loader for an OOD Dataset"):
            
            x = x.to(device)
            y = y.to(device)
            
            logits = self.get_logits(x, **kwargs)  # -> (batch_size, 19, H, W)
            
            max_logit = logits[:,:19,:,:].max(dim=1).values # eliminate background predicted logi
            
            mask = (y == 1)
            
            max_logits_ood.extend(max_logit[mask].cpu().tolist())
            max_logits_id.extend(max_logit[~mask].cpu().tolist())
        
        max_logits_ood = np.array(max_logits_ood)[::skip]
        max_logits_id = np.array(max_logits_id)[::skip]

        df = pd.DataFrame()

        labels = np.concatenate([
            np.array(["road_anomaly_ood"] * len(max_logits_ood)), 
            np.array(["road_anomaly_id"] * len(max_logits_id))
        ])

        df["logits"] = np.concatenate([max_logits_ood, max_logits_id]).astype(np.float32)
        df["labels"] = labels
        df["labels"] = df["labels"].astype('category')

        return df


    def extract_max_logits_id(
        self, 
        loader, 
        num_classes, 
        class_names, 
        device=torch.device('cpu'), 
        skip=10, 
        id_skip=100, 
        verbose=True, 
        **kwargs
    ):
        
        max_logits_id = []
        max_logits_per_class = [[] for _ in range(num_classes)]

        for x, y in tqdm(loader, desc='Cityscapes Val'):
    
            x = x.to(device)
            y = y.to(device)
            
            logits = self.get_logits(x, **kwargs)  # -> (batch_size, 19, H, W)
            
            max_logit = logits.max(dim=1).values

            max_logits_id.extend(max_logit.reshape(-1).cpu().tolist()[::id_skip])
            
            for c in range(num_classes):
                mask = (y == c)
                
                num_pixels = torch.sum(mask)
                
                if num_pixels == 0:
                    continue
                elif num_pixels < id_skip:
                    max_logits_per_class[c].extend(max_logit[mask].cpu().tolist())
                else:
                    max_logits_per_class[c].extend(max_logit[mask].cpu().tolist()[::id_skip])


        df = pd.DataFrame()
        df["logits"] = np.array(max_logits_id[::skip]).astype(np.float32)
        df["labels"] = np.array(['cityscapes_id'] * len(max_logits_id[::skip]))

        if verbose:
            print("Total Number of ID pixels", len(max_logits_id))
            print("Per Class Pixel Frequency:")

        for c in range(num_classes):
            
            class_name = class_names[c]
            
            if verbose:
                print("{:20} {:20}".format(class_name, len(max_logits_per_class[c])))
            
            class_df = pd.DataFrame()
            
            sk = 1
            if len(max_logits_per_class[c]) > 1e5:
                sk = 10
            
            class_df["logits"] = max_logits_per_class[c][::sk]
            class_df["labels"] = np.array([f'cityscapes_{class_name}'] * len(max_logits_per_class[c][::sk]))
            
            df = pd.concat([df, class_df], ignore_index=True)

        return df


    def visualize_kde(self, df, bw_adjust=1, chosen_cols=None):

        if chosen_cols is not None:
            mask = df['labels'].isin(chosen_cols)
            vis_df = df.loc[mask]
        else:
            vis_df = df

        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=vis_df, x='logits', hue='labels', fill=True, common_norm=False, bw_adjust=bw_adjust)
        plt.show()


class OODEvaluator:

    def __init__(
        self, 
        model: nn.Module, 
        ckpt_path: str, 
        model_ckpt_loader: Callable,
        inference_func: Callable,
        config: Optional[dict]
    ):
        
        self.model = model_ckpt_loader(model, ckpt_path, config)
        self.config = config
        self.inference_func = inference_func
        
    
    def get_logits(self, x, **kwargs):
        return self.inference_func(self.model, x, **kwargs)
    

    def calculate_ood_metrics(out, label):
    
        fpr, tpr, _ = roc_curve(label, out)

        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(label, out)
        prc_auc = average_precision_score(label, out)
        fpr = fpr_at_95_tpr(out, label)
        
        return roc_auc, prc_auc, fpr