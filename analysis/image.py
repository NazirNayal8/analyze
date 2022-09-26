import matplotlib.pyplot as plt
import torch
import numpy as np

from .metrics import find_tpr_threshold


def show_image(img, title=""):
    """
    Helper function for showing images
    """
    if isinstance(img, torch.Tensor):
        ready_img = img.clone()
        if len(ready_img.shape) == 3 and ready_img.shape[0] == 3:
            ready_img = ready_img.permute(1, 2, 0)
        ready_img = ready_img.cpu()

    elif isinstance(img, np.ndarray):
        ready_img = img.copy()
        if len(ready_img.shape) == 3 and ready_img.shape[0] == 3:
            ready_img = ready_img.transpose(1, 2, 0)
    else:
        raise ValueError(
            f"Unsupported type for image: ({type(img)}), only supports numpy arrays and Pytorch Tensors")
    print(type(ready_img), ready_img.shape)
    print(ready_img.dtype)
    plt.show(ready_img)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


def show_anomaly_map_without_void(map, ood_gts, title="Anomaly Score without Void", void_id=255, suppress_image=False):
    """
    Replaces Anomaly regions which should be ignored in evaluation with the lowest anomaly score
    """
    if isinstance(map, torch.Tensor):
        map = map.cpu().numpy()
    if isinstance(ood_gts, torch.Tensor):
        ood_gts = ood_gts.cpu().numpy()
    
    anomaly_map = map.copy()
    anomaly_map[ood_gts == void_id] = anomaly_map.min()

    if not suppress_image:
        show_image(anomaly_map, title=title)

    return anomaly_map

def show_anomaly_map_at_tpr_threshold(map, ood_gts, tpr_threshold, title="", void_id=255, suppress_image=False):
    """
    Shows and returns a discretized anomaly map where OOD pixels are chosen for the threshold that
    achieves tpr_threshold. For example, when tpr_threshold = 0.95, OOD pixels will be chosen for a 
    threshold that guarantees a true positive rate of 0.95
    """

    if tpr_threshold < 0 or tpr_threshold > 1:
        raise ValueError(f"TPR threshold must be between 0-1, given was: {tpr_threshold}")

    if isinstance(map, torch.Tensor):
        map = map.cpu().numpy()
    if isinstance(ood_gts, torch.Tensor):
        ood_gts = ood_gts.cpu().numpy()

    # find the threshold that achieves tpr_threshold true positive rate
    threshold = find_tpr_threshold(map, ood_gts, tpr_threshold)

    # OOD pixels
    anomaly_map = map.copy()
    ood_mask = (map > threshold) & (ood_gts != void_id)
    anomaly_map[ood_mask] = 255
    # inlier pixels
    anomaly_map[map < threshold] = 128
    # void pixels
    anomaly_map[ood_gts == void_id] = 0

    if not suppress_image:
        show_image(anomaly_map, title=title)

    return anomaly_map


    