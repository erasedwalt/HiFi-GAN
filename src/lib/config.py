import torch
from torch.utils.data import DataLoader

import json

from .model import HiFiGenerator, HiFiDiscriminator
from .loss import HiFiGeneratorLoss, HiFiDiscriminatorLoss
from .melspectrogram import MelSpectrogram
from .collator import LJSpeechCollator
from .dataset import LJSpeechDataset
from .logger import WandbLogger


class Config:
    def __init__(self, path):
        with open(path, 'r') as fp:
            self.config = json.load(fp)
        self.device = self.config['device']
        self.eval_interval = self.config['eval_interval']
        self.clip = self.config['clip']
        self.best_loss = self.config['best_loss']
        self.exp_name = self.config['exp_name']
        if len(self.config['chkpt']) > 0:
            self.to_load = torch.load(self.config['chkpt'], map_location='cpu')
        print(self.config)

    def get_models(self):
        generator = HiFiGenerator(**self.config['generator'])
        discriminator = HiFiDiscriminator()
        if len(self.config['chkpt']) > 0:
            generator.load_state_dict(self.to_load['gen'])
            discriminator.load_state_dict(self.to_load['discr'])
        if self.device == 'cpu':
            generator = generator.to('cpu')
            discriminator = discriminator.to('cpu')
        elif 'cuda' in self.device:
            generator = torch.nn.DataParallel(generator.to(self.device), device_ids=self.config['device_ids'])
            discriminator = torch.nn.DataParallel(discriminator.to(self.device), device_ids=self.config['device_ids'])
        print('Number of generator parameters', sum(p.numel() for p in generator.parameters()))
        print('Number of discriminator parameters', sum(p.numel() for p in discriminator.parameters()))
        return generator, discriminator

    def get_optimizers(self, gen, discr):
        optimizer_class = getattr(torch.optim, self.config['optim']['name'])
        optimizer_gen = optimizer_class(gen.parameters(), **self.config['optim']['args'])
        optimizer_discr = optimizer_class(discr.parameters(), **self.config['optim']['args'])
        if len(self.config['chkpt']) > 0:
            optimizer_gen.load_state_dict(self.to_load['optim_gen'])
            optimizer_discr.load_state_dict(self.to_load['optim_discr'])
        print(optimizer_gen)
        print(optimizer_discr)
        return optimizer_gen, optimizer_discr

    def get_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **self.config['scheduler'])
        print(scheduler)
        return scheduler

    def get_dataloader(self, train):
        melspec = MelSpectrogram(self.config['melspec'], for_loss=False)
        melspec_for_loss = MelSpectrogram(self.config['melspec'], for_loss=True)
        dataset = LJSpeechDataset(
            self.config['data']['train' if train else 'val']['path'],
            melspec,
            melspec_for_loss,
            train=train
        )
        loader = DataLoader(
            dataset,
            collate_fn=LJSpeechCollator(),
            **self.config['data']['train' if train else 'val']['args']
        )
        return loader

    def get_criterions(self):
        gen_loss = HiFiGeneratorLoss(**self.config['loss'])
        discr_loss = HiFiDiscriminatorLoss()
        return gen_loss, discr_loss

    def get_melspec(self):
        melspec = MelSpectrogram(self.config['melspec'], for_loss=True).to(self.device)
        return melspec

    def get_logger(self):
        if 'wandb' in self.config:
            logger = WandbLogger(**self.config['wandb'])
        else:
            logger = None
        return logger
