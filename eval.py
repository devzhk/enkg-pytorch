'''
This file implements evaluator for each problem.
'''

from abc import ABC, abstractmethod
import torch
import numpy
# from monai.metrics import PSNRMetric, SSIMMetric
from piq import LPIPS, psnr, ssim


class Evaluator(ABC):
    def __init__(self, metric_list):
        self.metric_list = metric_list
        self.metric_state = {key: 0.0 for key in metric_list.keys()}
        self.total = 0

    @abstractmethod
    def __call__(self, pred, target, observation=None):
        ''''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
            - observation (torch.Tensor): (N, *observation.shape) or (*observation.shape)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        pass

    def compute(self):
        '''
        Returns:
            - metric_state (Dict): a dictionary of metric values
        '''
        metric_state = {key: val / self.total for key, val in self.metric_state.items()}
        return metric_state


def relative_l2(pred, target):
    ''''
    Args:
        - pred (torch.Tensor): (N, C, H, W)
        - target (torch.Tensor): (C, H, W)
    Returns:
        - rel_l2 (torch.Tensor): (N,), relative L2 error
    '''
    diff = pred - target
    l2_norm = torch.linalg.norm(target.reshape(-1))
    rel_l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / l2_norm
    return rel_l2


class NavierStokes2d(Evaluator):
    def __init__(self, ):
        metric_list = {'relative l2': relative_l2}
        super(NavierStokes2d, self).__init__(metric_list)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        self.total += 1
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            if len(target.shape) == 3:
                val = metric_func(pred, target).item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name] += val
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name] += val
        return metric_dict


class Image(Evaluator):
    def __init__(self, ):
        self.eval_batch = 32
        metric_list = {'psnr': lambda x, y: psnr(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
                       'ssim': lambda x, y: ssim(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
                       'lpips': LPIPS(replace_pooling=True, reduction='none')}
        super(Image, self).__init__(metric_list)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        self.total += 1
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            if pred.shape != target.shape:
                num_batches = pred.shape[0] // self.eval_batch
                for i in range(num_batches):
                    pred_batch = pred[i * self.eval_batch: (i + 1) * self.eval_batch]
                    target_batch = target.repeat(pred_batch.shape[0], 1, 1, 1)
                    val = metric_func(pred_batch, target_batch).squeeze(-1).sum()
                    metric_dict[metric_name] += val
                metric_dict[metric_name] = metric_dict[metric_name] / pred.shape[0]
                self.metric_state[metric_name] += metric_dict[metric_name]
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name] += val
        return metric_dict
    
    
class MRI(Evaluator):
    def __init__(self,):
        self.eval_batch = 32
        metric_list = {'psnr': lambda x, y: psnr(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
                       'ssim': lambda x, y: ssim(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none')}
        super(MRI, self).__init__(metric_list)

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        self.total += 1
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            if pred.shape != target.shape:
                num_batches = pred.shape[0] // self.eval_batch
                for i in range(num_batches):
                    pred_batch = pred[i * self.eval_batch: (i + 1) * self.eval_batch]
                    target_batch = target.repeat(pred_batch.shape[0], 1, 1, 1)
                    val = metric_func(pred_batch, target_batch).squeeze(-1).sum()
                    metric_dict[metric_name] += val
                metric_dict[metric_name] = metric_dict[metric_name] / pred.shape[0]
                self.metric_state[metric_name] += metric_dict[metric_name]
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name] += val
        return metric_dict
    


