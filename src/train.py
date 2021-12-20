import torch
import numpy as np

import argparse
from tqdm import tqdm
import random

from lib import Config


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def train(
    generator,
    discriminator,
    optimizer_gen,
    optimizer_discr,
    scheduler_gen,
    scheduler_discr,
    criterion_gen,
    criterion_discr,
    train_loader,
    val_loader,
    melspec_computer,
    eval_interval,
    logger,
    device,
    best_loss,
    clip,
    exp_name
):
    global_steps = 0
    generator.train()
    discriminator.train()
    while True:
        for i, (waveform, melspec, melspec_for_loss) in enumerate(tqdm(train_loader)):
            optimizer_gen.zero_grad()
            optimizer_discr.zero_grad()

            waveform = waveform.to(device)
            melspec = melspec.to(device)
            melspec_for_loss = melspec_for_loss.to(device)

            gen_waveform = generator(melspec)
            preds_generated, preds_true, _, _ = \
                discriminator(gen_waveform.detach(), waveform[:, None, :])

            # Discriminator part
            loss_discr = criterion_discr.forward(preds_generated, preds_true)
            loss_discr.backward()
            # grad_discr = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip)
            optimizer_discr.step()

            preds_generated, preds_true, fmaps_generated, fmaps_true = \
                discriminator(gen_waveform, waveform[:, None, :])
            gen_melspec = melspec_computer(gen_waveform.squeeze(1))

            # Generator part
            loss_gen = criterion_gen.forward(
                preds_generated + preds_true,
                gen_melspec,
                melspec_for_loss,
                fmaps_generated,
                fmaps_true
            )

            loss_gen.backward()
            # grad_gen = torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
            optimizer_gen.step()
            global_steps += 1

            lr = scheduler_gen.get_last_lr()[0]

            to_log = {'Train Gen Loss': loss_gen.item(),
                      'Train Discr Loss': loss_discr.item(),
                      # 'Grad Discr': grad_discr.item(),
                      # 'Grad Gen': grad_gen.item(),
                      'LR': lr}

            if logger is None:
                print(to_log)
            else:
                logger.log_metrics(to_log)

            if global_steps % eval_interval == 0:
                best_loss = evaluate(
                    generator,
                    discriminator,
                    optimizer_gen,
                    optimizer_discr,
                    val_loader,
                    melspec_computer,
                    logger,
                    device,
                    best_loss,
                    exp_name
                )
                generator.train()
                discriminator.train()

        scheduler_gen.step()
        scheduler_discr.step()



def evaluate(
    generator,
    discriminator,
    optimizer_gen,
    optimizer_discr,
    val_loader,
    melspec_computer,
    logger,
    device,
    best_loss,
    exp_name
):
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        global_loss = 0.
        for i, (waveform, melspec, melspec_for_loss) in enumerate(tqdm(val_loader)):
            waveform = waveform.to(device)
            melspec = melspec.to(device)
            melspec_for_loss = melspec_for_loss.to(device)

            gen_waveform = generator(melspec)
            gen_melspec = melspec_computer(gen_waveform.squeeze(1))

            loss_val = torch.mean(torch.abs(gen_melspec - melspec_for_loss))
            global_loss += loss_val.item()

        rand_int = random.randint(0, waveform.shape[0] - 1)
        log_true_wav = waveform[rand_int].detach().cpu().squeeze()
        log_gen_wav = gen_waveform[rand_int].detach().cpu().squeeze()
        log_true_melspec = melspec_for_loss[rand_int].detach().cpu()
        log_gen_melspec = gen_melspec[rand_int].detach().cpu()

        if logger is not None:
            logger.log_spec_and_audio(
                log_gen_melspec,
                log_true_melspec,
                log_gen_wav,
                log_true_wav
            )

        global_loss /= len(val_loader)

        to_save = {'gen': generator.module.state_dict(),
                   'discr': discriminator.module.state_dict(),
                   'optim_discr': optimizer_discr.state_dict(),
                   'optim_gen': optimizer_gen.state_dict()}

        if best_loss > global_loss:
            best_loss = global_loss
            torch.save(to_save, f'../chkpt/{exp_name}_best.pt')
        torch.save(to_save, f'../chkpt/{exp_name}_last.pt')

        to_log = {'Val Loss': global_loss, 'Best Val Loss': best_loss}
        if logger is not None:
            logger.log_metrics(to_log)
        else:
            print(to_log)
    return best_loss


def _parse_args():
    parser = argparse.ArgumentParser(description='Train argparser')
    parser.add_argument(
        '-c', '--config',
        help='Path to config',
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    config = Config(args.config)

    generator, discriminator = config.get_models()
    melspec = config.get_melspec()
    gen_loss, discr_loss = config.get_criterions()
    train_loader = config.get_dataloader(True)
    val_loader = config.get_dataloader(False)
    optimizer_gen, optimizer_discr = config.get_optimizers(generator, discriminator)
    scheduler_gen = config.get_scheduler(optimizer_gen)
    scheduler_discr = config.get_scheduler(optimizer_discr)
    logger = config.get_logger()

    train(generator, discriminator, optimizer_gen, optimizer_discr,
          scheduler_gen, scheduler_discr, gen_loss, discr_loss,
          train_loader, val_loader, melspec, config.eval_interval,
          logger, config.device, config.best_loss, config.clip,
          config.exp_name)
