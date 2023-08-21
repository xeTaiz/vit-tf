# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self,
                 base_encoder,
                 dim,
                 encoder_kwargs={'num_classes': 128},
                 K=65536,
                 m=0.999,
                 T=0.07,
                 samples_to_enqueue=256):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.F = dim
        self.samples_to_enqueue = samples_to_enqueue

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(**encoder_kwargs)
        self.encoder_k = base_encoder(**encoder_kwargs)

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        print(keys.shape)
        batch_size = keys.shape[1]  # TODO: this used to key keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr +
                   batch_size] = keys  #.T   TODO: this was transposed. maybe the deactivated ddp stuff is related
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward_old(self, im_q, im_k, correspondence_q, correspondence_k):
        """
        Input:
            im_q: a batch of query volumes (B, C, D ,H, W)
            im_k: a batch of key volumes (B, C, D ,H, W)
            correspondences: dict of classname: (N, )
        Output:
            logits, targets
        """

        # compute query features
        features = self.encoder_q(im_q)  # queries: NxFxHxW
        q = nn.functional.normalize(features, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxFxDxHxW
            k = nn.functional.normalize(k, dim=1)

            # Sample corresponding features in k, interpolate, permute to NxHxWxF to keep features when masking
            k = F.grid_sample(k, correspondence_k[None, :, None, None],
                              mode='nearest').squeeze()  # BxFxS

        q = F.grid_sample(q, correspondence_q[None, :, None, None],
                          mode='bilinear').squeeze()  # BxFxS

        print('q', q.shape)
        print('k', k.shape)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: BxS
        l_pos = torch.einsum('bfs,bfs->bs', [q, k]).unsqueeze(-1)
        # negative logits: BxK
        l_neg = torch.einsum('bfs,fk->bsk', [q, self.queue.clone().detach()])

        # logits: BxSx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=-1).reshape(-1, 1 + self.K)
        print('logits', logits.shape)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0],
                             dtype=torch.long,
                             device=im_q.device)

        # dequeue and enqueue   BxFxS -> FxBxS  ->  Fx(B*S)
        to_enq = k.permute(1, 0, 2).reshape(self.F, -1)
        print('to_enq', to_enq.shape)
        self._dequeue_and_enqueue(to_enq)

        return logits, labels, features

    def forward(self, im):
        """
        Input:
            im: a batch of query volumes (B, C, D ,H, W)
        Output:
            logits
        """

        # compute query features
        features = self.encoder_q(im)  # queries: NxFxHxW
        q = nn.functional.normalize(features, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im)  # keys: NxFxDxHxW
            k = nn.functional.normalize(k, dim=1)
        return q, k
