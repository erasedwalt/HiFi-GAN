import torch
from torch import nn


class HiFiGeneratorLoss(nn.Module):
    '''
    HiFi Generator Loss
    '''
    def __init__(self, lambda_fm=2., lambda_mel=45.):
        super(HiFiGeneratorLoss).__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def forward(
        self,
        discr_preds_generated,
        spec_generated,
        spec_true,
        fmaps_generated,
        fmaps_true
    ):

        loss_adv = 0.
        for i in range(len(discr_preds_generated)):
            loss_adv += torch.mean(((discr_preds_generated[i] - 1.) ** 2))

        loss_fm = 0.
        for i in range(len(fmaps_generated)):
            loss_fm += torch.mean(torch.abs(fmaps_generated[i] - fmaps_true[i]) \
                                  / fmaps_true[i].shape[1])

        loss_mel = torch.mean(torch.abs(spec_generated - spec_true))

        loss = loss_adv + self.lambda_fm * loss_fm + self.lambda_mel * loss_mel

        return loss


class HiFiDiscriminatorLoss(nn.Module):
    '''
    HiFi Diiscriminator Loss
    '''
    def __init__(self):
        super(HiFiDiscriminatorLoss, self).__init__()

    def forward(
        self,
        discr_preds_generated,
        discr_preds_true
    ):
        loss = 0.
        for i in range(len(discr_preds_generated)):
            loss += torch.mean((discr_preds_true[i] - 1.) ** 2 + \
                                   discr_preds_generated[i] ** 2)
        return loss
