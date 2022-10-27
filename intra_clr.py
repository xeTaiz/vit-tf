import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraCLR(nn.Module):
    def __init__(self, base_encoder, encoder_kwargs={}, neg_samples=40000, T=0.07):
        ''' Initializes Contrastive loss within a feature volume

        Args:
            base_encoder (nn.Module): Encoder producing a feature map with spatial dimensions
            encoder_kwargs (dict, optional): Arguments to pass to the encoder. Defaults to {}.
            neg_samples (int, optional): Number of negative samples to pick from the other classes. Defaults to 40000.
            T (float, optional): Temperature for InfoNCE loss. Defaults to 0.07.
        '''
        self.neg_samples = neg_samples
        self.encoder = base_encoder(**encoder_kwargs)

    def forward(self, vol, correspondences):
        ''' Computes InfoNCE in between voxels of different classes and

        Args:
            vol (Tensor): Input volume (BS, C, D, H, W)
            correspondences (dict): Dict of classname: Tensor indices (N, 3)

        Returns:
            logits (Tensor): Logits of the feature matches
            labels (Tensor): Labels describing which of the logits match pos and neg
            features (Tensor): Feature volume of the encoder
        '''
        features = self.encoder(vol)
        q = F.normalize(features, dim=1)

        pos_groups = { k: F.grid_sample(features, idx[None, :, None, None], 
            mode='trilinear') for k, idx in correspondences.items() }

            