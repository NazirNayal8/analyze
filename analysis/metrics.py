import numpy as np
from sklearn.metrics import roc_curve

def find_tpr_threshold(anomaly_score, ood_gts, threshold=0.95):
    
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_score[ood_mask]
    ind_out = anomaly_score[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    fpr, tpr, thresholds = roc_curve(val_label, val_out)
    fpr95_threshold = np.where(tpr > threshold)[0].min()
    
    return thresholds[fpr95_threshold]