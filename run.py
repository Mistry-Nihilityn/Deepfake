import datetime
import os
import random
import yaml
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from dataset.AbstractDataset import AbstractDataset
from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR
from prepare_data import load_train_val, load_test
from trainer.tester import Tester

from trainer.trainer import Trainer
from detectors import DETECTOR
from logger import create_logger


def init_seed(config):
    if config['seed'] is None:
        config['seed'] = random.randint(1, 10000)
    random.seed(config['seed'])
    if config['cuda']:
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])


def prepare_train_data(config):
    train_files, val_files = load_train_val(config)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=AbstractDataset(train_files, config, mode='train'),
        batch_size=config["dataset"]["train"]['batch_size'],
        shuffle=True,
        num_workers=int(config["dataset"]["train"]['workers'])
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=AbstractDataset(val_files, config, mode='val'),
        batch_size=config["dataset"]["train"]['batch_size'],
        shuffle=False,
        num_workers=int(config["dataset"]["train"]['workers'])
    )
    return train_data_loader, val_data_loader


def prepare_test_data(config):
    test_files = load_test(config)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=AbstractDataset(test_files, config, mode='test'),
        batch_size=config["dataset"]["test"]['batch_size'],
        shuffle=True,
        num_workers=int(config["dataset"]["test"]['workers'])
    )
    return train_data_loader


def choose_optimizer(model, config):
    opt_name = config['optimizer']['base']
    add_name = config['optimizer']['additional']
    base_optimizer_class = None
    if opt_name == 'sgd':
        base_optimizer_class = optim.SGD
    elif opt_name == 'adam':
        base_optimizer_class = optim.Adam

    if add_name == 'sam':
        optimizer = SAM(
            model.parameters(),
            base_optimizer_class,
            **config['optimizer'][add_name],
            **config['optimizer'][opt_name]
        )
    else:
        optimizer = base_optimizer_class(
            model.parameters(),
            **config['optimizer'][opt_name]
        )
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['n_epochs'],
            int(config['n_epochs'] / 4),
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def get_weight(data_loader: DataLoader):
    fake_cnt = data_loader.dataset.fake_cnt
    real_cnt = data_loader.dataset.real_cnt
    return [real_cnt / min(fake_cnt, real_cnt),
            fake_cnt / min(fake_cnt, real_cnt)]


def train(config):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join(config['log_dir'],config['model_name'], config['dataset']["train"]["name"], time_now)
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, 'training.log'))
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    train_data_loader, val_data_loader = prepare_train_data(config)
    logger.info(f"Train set:")
    logger.info(f"  - fake: {train_data_loader.dataset.fake_cnt:,}")
    logger.info(f"  - real: {train_data_loader.dataset.real_cnt:,}")
    logger.info(f"  - total: {train_data_loader.dataset.fake_cnt + train_data_loader.dataset.real_cnt:,}")
    logger.info(f"\nValidation set:")
    logger.info(f"  - fake: {val_data_loader.dataset.fake_cnt:,}")
    logger.info(f"  - real: {val_data_loader.dataset.real_cnt:,}")
    logger.info(f"  - total: {val_data_loader.dataset.fake_cnt + val_data_loader.dataset.real_cnt:,}")

    config["dataset"]["weight"] = get_weight(train_data_loader)
    logger.info(f"Weight: {config['dataset']['weight']}")

    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    trainer = Trainer(config, model, optimizer, scheduler, logger, log_dir=log_dir)
    # start training
    for epoch in range(max(config['start_epoch'], 1), config['n_epochs'] + 1):
        trainer.model.epoch = epoch
        metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            val_data_loaders=val_data_loader
        )
        if metric is not None:
            loss = metric["avg_loss"]
            acc = metric["acc"]
            logger.info(f"===> Epoch[{epoch}] end with testing acc:{acc}, loss:{loss}!")
        if scheduler is not None:
            scheduler.step()

    metric = trainer.best_metric
    epoch = metric['epoch']
    loss = metric["avg_loss"]
    acc = metric["acc"]
    logger.info(f"Stop Training on best Testing metric epoch:{epoch}, acc:{acc}, loss:{loss}")

    return log_dir


def test(config, train_dir):
    log_dir = os.path.join(train_dir, config["dataset"]["test"]["name"])
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, 'test.log'))
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    test_data_loader = prepare_test_data(config)

    logger.info(f"Train set:")
    logger.info(f"  - fake: {test_data_loader.dataset.fake_cnt:,}")
    logger.info(f"  - real: {test_data_loader.dataset.real_cnt:,}")
    logger.info(f"  - total: {test_data_loader.dataset.fake_cnt + test_data_loader.dataset.real_cnt:,}")

    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    model.load_state_dict(torch.load(os.path.join(train_dir, 'best_model.pth')), strict=False)

    tester = Tester(config, model, logger, log_dir=log_dir)
    tester.test(test_data_loader)

    logger.info(f"Test finished!")

def run(config_path):
    # parse options and load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["dataset"] = {}
    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True

    for train_dataset_config_path in config["train_dataset_configs"]:
        with open(train_dataset_config_path, 'r') as f:
            config["dataset"]["train"] = yaml.safe_load(f)
        train_dir = train(config)
        for test_dataset_config_path in config["test_dataset_configs"]:
            with open(test_dataset_config_path, 'r') as f:
                config["dataset"]["test"] = yaml.safe_load(f)
            test(config, train_dir)
        
if __name__ == '__main__':
    run("config/run_config.yaml")
