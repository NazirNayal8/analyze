import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable
from tqdm import tqdm
from torchmetrics import JaccardIndex


class SemSegAnalyzer:

    def __init__(
        self,
        model: nn.Module,
        inference_func: Callable,
    ) -> None:

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

        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers)
        mIoU_metric = JaccardIndex(
            task='multiclass', num_classes=num_classes, ignore_index=ignore_index).to(device)

        for x, y in tqdm(loader):

            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():

                logits = self.get_logits(x, **kwargs)

            dummy_extension = torch.zeros(
                logits.shape[0], 1, logits.shape[2], logits.shape[3]).to(device)
            logits = torch.cat([logits, dummy_extension], dim=1)

            mIoU_metric.update(logits, y)

        return mIoU_metric.cpu().compute()
