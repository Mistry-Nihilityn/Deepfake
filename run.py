import datetime
import os
import random
import sys
import time
import traceback

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import yaml
from torch.utils.data import DataLoader

from dataset.DeepfakeDataset import TrainDataset, TestDataset
from detectors import DETECTOR
from logger import create_logger, close_logger
from optimizor.LinearLR import LinearDecayLR
from optimizor.SAM import SAM
from prepare_data import load_train, load_test
from trainer.tester import Tester
from trainer.trainer import Trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['seed'] is None:
        config['seed'] = random.randint(1, 10000)
    random.seed(config['seed'])
    if config['cuda']:
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])


def get_length(files, label):
    return sum(list(map(lambda x: len(x[label]), files.values())))


def prepare_train_data(config, logger):
    train_files, val_files, test_files = load_train(config, logger)
    data_config = config["dataset"]["train"]

    logger.info(f"{'Preparing Data Loader':=^50}")
    logger.info(f"Train: {get_length(train_files, 'real')} real, {get_length(train_files, 'fake')} fake.")
    logger.info(f"Validate: {get_length(val_files, 'real')} real, {get_length(val_files, 'fake')} fake.")
    logger.info(f"Test: {get_length(test_files, 'real')} real, {get_length(test_files, 'fake')} fake.")

    train_data_loader = torch.utils.data.DataLoader(
            dataset=TrainDataset(
                train_files,
                resolution=config['resolution'],
                balance=data_config['balance'],
                sample_per_class=data_config['sample_per_class'],
                augment_config=config['data_aug'] if config['use_data_augmentation'] else None,
                mean=config['mean'],
                std=config['mean']
            ),
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=int(data_config['workers'])
        )
    val_data_loader = torch.utils.data.DataLoader(
            dataset=TrainDataset(
                val_files,
                resolution=config['resolution'],
                balance=data_config['balance'],
                sample_per_class=int(data_config['sample_per_class']*data_config['split']['val']//data_config["split"]['train']),
                augment_config=config['data_aug'] if config['use_data_augmentation'] else None,
                mean=config['mean'],
                std=config['mean']
            ),
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=int(data_config['workers'])
        )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=TestDataset(
            test_files,
            resolution=config['resolution'],
            balance=data_config['balance'],
            augment_config=None,
            mean=config['mean'],
            std=config['mean']
        ),
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=int(data_config['workers'])
    ) if len(test_files) > 0 else None
    return train_data_loader, val_data_loader, test_data_loader


def prepare_test_data(config, logger):
    data_config = config["dataset"]["test"]
    test_files = load_test(config, logger)

    logger.info(f"{'Preparing Data Loader':=^50}")
    logger.info(f"Test: {get_length(test_files, 'real')} real, {get_length(test_files, 'fake')} fake.")
    test_data_loader = torch.utils.data.DataLoader(
        dataset=TestDataset(
            test_files,
            resolution=config['resolution'],
            balance=data_config['balance'],
            augment_config=None,
            mean=config['mean'],
            std=config['mean']
        ),
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=int(data_config['workers'])
    )
    return test_data_loader


def choose_optimizer(model, config):
    opt_name = config['optimizer']['base']
    add_name = config['optimizer']['additional']
    base_optimizer_class = None
    if opt_name == 'sgd':
        base_optimizer_class = optim.SGD
    elif opt_name == 'adam':
        base_optimizer_class = optim.Adam

    if add_name == 'sam' or add_name == 'sam+is-sam':
        optimizer = SAM(
            model.feature_params(),
            model.classifier_params(),
            base_optimizer_class,
            config['optimizer']['sam']["rho"],
            config['optimizer']['sam']["affect_classifier"],
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
    if min(fake_cnt, real_cnt):
        return [(real_cnt + fake_cnt) / real_cnt,
                (real_cnt + fake_cnt) / fake_cnt]
    else:
        return [1., 1.]


def train(config):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join(config['log_dir'], config['model_name'], config['dataset']["train"]["name"], time_now)
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, 'training.log'))
    logger.info("--------------- Global Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    train_data_loader, val_data_loader, test_data_loader = prepare_train_data(config, logger)
    logger.info("--------------- Train Configuration ---------------")
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
    model = model_class(config).to(device)

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
    close_logger(logger)

    if config["dataset"]["train"]["type"] == "in-domain":
        logger.info("--------------- In-domain Test Configuration ---------------")
        logger.info(f"Test set:")
        logger.info(f"  - fake: {test_data_loader.dataset.fake_cnt:,}")
        logger.info(f"  - real: {test_data_loader.dataset.real_cnt:,}")
        logger.info(f"  - total: {test_data_loader.dataset.fake_cnt + test_data_loader.dataset.real_cnt:,}")

        tester = Tester(config, model, logger, log_dir=os.path.join(log_dir, 'in-domain'))
        tester.test(test_data_loader)
        logger.info(f"Test finished!")

    close_logger(logger)
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

    test_data_loader = prepare_test_data(config, logger)

    logger.info(f"Train set:")
    logger.info(f"  - fake: {test_data_loader.dataset.fake_cnt:,}")
    logger.info(f"  - real: {test_data_loader.dataset.real_cnt:,}")
    logger.info(f"  - total: {test_data_loader.dataset.fake_cnt + test_data_loader.dataset.real_cnt:,}")

    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    model.load_state_dict(torch.load(os.path.join(train_dir, 'best_model.pth')), strict=False)
    model = model.to(device)
    tester = Tester(config, model, logger, log_dir=log_dir)
    tester.test(test_data_loader)

    logger.info(f"Test finished!")
    close_logger(logger)


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
            config["dataset"]["train"] = config["dataset"]["test"] = yaml.safe_load(f)
        train_dir = train(config)
        if config["dataset"]["train"]["type"] == "cross-domain":
            for test_dataset_config_path in config["test_dataset_configs"]:
                with open(test_dataset_config_path, 'r') as f:
                    config["dataset"]["test"] = yaml.safe_load(f)
                test(config, train_dir)


RUNS = [
    # "config/run_coatnet.yaml",
    # "config/run_coatnet_sam.yaml",
    # "config/run_coatnet_is_sam.yaml",
    # "config/run_coatnet_sam_is_sam.yaml",
    "config/run_resnet50.yaml",
    # "config/run_resnet50_sam.yaml",
    # "config/run_resnet50_is_sam.yaml",
    # "config/run_resnet50_sam_is_sam.yaml"
    # "config/run_dino.yaml",
]

if __name__ == '__main__':
    with open("./log/errors.log", 'a') as f:
        for run_config in RUNS:
            f.write(f"Runing {run_config} at {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
            try:
                run(run_config)
            except Exception as e:
                print("Error!")
                traceback.print_tb(e.__traceback__, file=f)
                # raise e
        f.write("Finished!")
        print("Finished!")
    os.system("/usr/bin/shutdown")
