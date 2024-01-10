import torch
import random
import numpy as np
import math
from easydict import EasyDict as edict

from mix import DiffusionMixture

from losses import get_loss_fn_e3nn
from losses_cdr import get_loss_fn_e3nn as loss_fn_cdrh3
from solver import get_pc_sampler
from utils.ema import ExponentialMovingAverage

from prior_drift import prior_none

def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model(params):
    params_ = params.copy()
    model = ScoreModel(**params_)
    return model


def load_model_optimizer(params, config_train, device, resume_state_dict=None):
    model = load_model(params)
    if resume_state_dict is not None:
        if 'module.' in list(resume_state_dict.keys())[0]:
            # strip 'module.' at front; for DataParallel models
            resume_state_dict = {k[7:]: v for k, v in resume_state_dict.items()}
        model.load_state_dict(resume_state_dict)

    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')

    if config_train.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr, 
                                    weight_decay=config_train.weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr, 
                                    weight_decay=config_train.weight_decay)
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, sample=False, ddp=False, world_size=1, rank=0):
    from dataset import get_dataloader
    return get_dataloader(config, sample=sample, ddp=ddp)

def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b


def load_mix(config_mix):
    mix_type = config_mix.type
    eta = config_mix.eta
    drift_coeff = config_mix.drift_coeff
    schedule = config_mix.schedule
    sigma_0 = config_mix.sigma_0
    sigma_1 = config_mix.sigma_1
    num_scales = config_mix.num_scales
    mix = DiffusionMixture(bridge=mix_type, eta=eta, drift_coeff=drift_coeff, schedule=schedule, 
                            sigma_0=sigma_0, sigma_1=sigma_1, N=num_scales)
    return mix

def load_prior(config_prior, config):
    prior_type = config_prior.type
    if prior_type == 'None':
        prior = prior_none()
    elif prior_type == 'eig':
        prior = prior_eig(config_prior.scale)
    elif prior_type == 'spec':
        prior = prior_spectrum(config_prior.scale, load_spectrum(config))
    else:
        raise NotImplementedError(f"Prior class {prior_type} not yet supported.")
    return prior

def load_e3nn_loss_fn(config):
    mix_atom = load_mix(config.mix.aa)
    mix_pos = load_mix(config.mix.pos)

    if config.model.type == 'cdrh3':
        get_loss_fn = loss_fn_cdrh3
    else:
        get_loss_fn = get_loss_fn_e3nn

    loss_fn = get_loss_fn(mix_atom, mix_pos,
                        reduce_mean=config.train.reduce_mean, eps=config.train.eps, loss_type=config.train.loss_type)

    return loss_fn

def load_model_params(config):
    return config.model


def load_ckpt(config, device, ts=None, return_ckpt=False):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    path = f'checkpoints/{config.data.data}/{config.ckpt}.pth'
    ckpt_dict = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model


def load_opt_from_ckpt(config_train, state_dict, model):
    if config_train.optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr, 
                                        weight_decay=config_train.weight_decay)
    elif config_train.optim=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr, amsgrad=True,
                                        weight_decay=config_train.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer:{config_train.optim} not implemented.')
    optimizer.load_state_dict(state_dict)
    return optimizer

def load_sampling_fn(config_train, config_sampler, config_sample, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    get_sampler = get_pc_sampler

    mix = edict({'aa': load_mix(config_train.mix.aa), 'pos': load_mix(config_train.mix.pos)})
    prior = edict({'aa': load_prior(config_sampler.prior.aa, config_train),
                   'pos': load_prior(config_sampler.prior.pos, config_train)})

    sampling_fn = get_sampler(mix=mix, prior=prior, sampler=config_sampler,
                                denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, device=device_id)
    return sampling_fn