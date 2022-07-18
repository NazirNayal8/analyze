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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from ood_metrics import fpr_at_95_tpr

from .utils import unnormalize_tensor


class MaxLogitsAnalyzer:

    def __init__(
        self, 
        model: nn.Module, 
        inference_func: Callable,
    ):
        
        self.model = model
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

    def extract_max_logits_ood(self, loader, dataset_name, device=torch.device('cpu'), skip=1, **kwargs):

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
            np.array([f"{dataset_name}_ood"] * len(max_logits_ood)), 
            np.array([f"{dataset_name}_id"] * len(max_logits_id))
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
        dataset_name,
        device=torch.device('cpu'), 
        skip=10, 
        id_skip=100, 
        verbose=True, 
        **kwargs
    ):
        
        max_logits_id = []
        max_logits_per_class = [[] for _ in range(num_classes)]

        for x, y in tqdm(loader, desc='Loader for an ID Dataset'):
    
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
        df["labels"] = np.array([f'{dataset_name}_id'] * len(max_logits_id[::skip]))

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
            class_df["labels"] = np.array([f'{dataset_name}_{class_name}'] * len(max_logits_per_class[c][::sk]))
            
            df = pd.concat([df, class_df], ignore_index=True)

        return df


    def visualize_kde(self, df, bw_adjust=1, figsize=(8,5), chosen_cols=None):

        if chosen_cols is not None:
            mask = df['labels'].isin(chosen_cols)
            vis_df = df.loc[mask]
        else:
            vis_df = df

        plt.figure(figsize=figsize)
        sns.kdeplot(data=vis_df, x='logits', hue='labels', fill=True, common_norm=False, bw_adjust=bw_adjust)
        plt.show()


class OODEvaluator:

    def __init__(
        self, 
        model: nn.Module, 
        inference_func: Callable,
    ):
        
        self.model = model
        self.inference_func = inference_func
        
    
    def get_logits(self, x, **kwargs):
        return self.inference_func(self.model, x, **kwargs)
    

    def calculate_ood_metrics(self, out, label):
    
        fpr, tpr, _ = roc_curve(label, out)

        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(label, out)
        prc_auc = average_precision_score(label, out)
        fpr = fpr_at_95_tpr(out, label)
        
        return roc_auc, prc_auc, fpr

    def evaluate_ood(self, anomaly_score, ood_gts, verbose=True):

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_score[ood_mask]
        ind_out = anomaly_score[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        auroc, aupr, fpr = self.calculate_ood_metrics(val_out, val_label)

        if verbose:
            print(f'Max Logits: AUROC score: {auroc}')
            print(f'Max Logits: AUPRC score: {aupr}')
            print(f'Max Logits: FPR@TPR95: {fpr}')

        result = {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr
        }

        return result

    def compute_max_logit_scores(self, loader, device=torch.device('cpu'), return_preds=False, upper_limit=2000):
    
        anomaly_score = []
        ood_gts = []
        predictions = []
        jj = 0
        for x, y in tqdm(loader, desc="Dataset Iteration"):
            
            if jj >= upper_limit:
                break
            jj += 1
            
            x = x.to(device)
            y = y.to(device)

            ood_gts.extend([y.cpu().numpy()])

            logits = self.get_logits(x)  # -> (batch_size, 19, H, W)

            max_logit, preds = logits[:,:19,:,:].max(dim=1) # eliminate background predicted logit
            
            if return_preds:
                predictions.extend([preds.cpu().numpy()])
            
            anomaly_score.extend([-max_logit.cpu().numpy()])

        ood_gts = np.array(ood_gts)
        anomaly_score = np.array(anomaly_score)
        
        if return_preds:
            predictions = np.array(predictions)
            return anomaly_score, ood_gts, predictions 
        
        return anomaly_score, ood_gts

    def log_to_wandb(
        self, 
        project_name, 
        run_name, 
        imgs, 
        ood_gts, 
        preds,
        anomaly_score,
        class_names,
        metrics=None,
    ):
        
        logger = wandb.init(project=project_name, name=run_name)

        self.log_anomaly_maps(imgs, anomaly_score, ood_gts, logger=logger)
        self.log_id_maps(imgs, preds, class_names, y=None, logger=logger)
        
        if metrics is not None:
            logger.log({
                    'OOD_test/AUROC': metrics['auroc'],
                    'OOD_test/AUPR': metrics['aupr'],
                    'OOD_test/FPR95': metrics['fpr95'],
                })

        wandb.finish()

    def log_id_maps(self, imgs, preds, class_names, logger, y=None, upper_limit=100):
    
        if isinstance(class_names, list):
            class_labels = {}
            for i in range(len(class_names)):
                class_labels[i] = class_names[i]
        else:
            class_labels = class_names
        
        num_samples = imgs.shape[0]
        seg_logs = []
        for i in range(num_samples):
            
            if i >= upper_limit:
                break
            
            mask = {
                'predictions': {
                    'mask_data': preds[i].squeeze(),
                    'class_labels': class_labels
                },
            }
            if y is not None:
                mask['ground_truth'] = {
                    'mask_data': y[i],
                    'class_labels': class_labels
                }
            
            seg_logs.extend([wandb.Image(imgs[i], masks=mask)])
        
        logger.log({'ID/predictions': seg_logs})

    def log_anomaly_maps(self, imgs, scores, y, logger, threshold=0.5, upper_limit=100):
    
        scores_min = scores.min()
        scores_max = scores.max()

        scores_norm = (scores - scores_min) / (scores_max - scores_min) 

        predictions = np.zeros_like(scores_norm)
        predictions[scores_norm > threshold] = 1

        num_samples = scores.shape[0]    
        
        novel_logs = []
        novel_table = wandb.Table(columns=['ID', 'Image'])

        for i in range(num_samples):

            if i >= upper_limit:
                break

            novel_mask = {
                'predictions': {
                    'mask_data': predictions[i].squeeze(),
                    'class_labels': {0: 'ID', 1: 'Novel'}
                },
                'ground_truth': {
                    'mask_data': y[i].squeeze(),
                    'class_labels': {0: 'ID', 1: 'Novel', 255: 'background'}
                }
            }
            wandb_image = wandb.Image(imgs[i].squeeze(), masks=novel_mask)
            
            novel_logs.extend([wandb_image])
            novel_table.add_data(i, wandb_image)

            fig = plt.figure(constrained_layout=True, figsize=(20, 14))

            ax = fig.subplot_mosaic(
                [['image', 'score']]
            )

            ax['image'].imshow(imgs[i].squeeze())
            ax['image'].set_title('Original Image')

            ax['score'].imshow(scores_norm[i].squeeze())
            ax['score'].set_title('Anomaly Scores')

            logger.log({
                f'OOD_P_MAPS/image_{i}': plt
            })

        logger.log({
            'tables/predictions_ood': novel_table,
            'OOD/seg_vis': novel_logs
        })
    
    def get_imgs(self, dataset, image_mean, image_std, out_type=np.uint8):

        imgs = []
        for i in range(len(dataset)):
            x, _ = dataset[i]
            imgs.extend([unnormalize_tensor(x.float(), np.array(image_mean), np.array(image_std)).cpu().permute(1, 2, 0).numpy()])

        imgs = np.array(imgs).astype(out_type)

        return imgs
