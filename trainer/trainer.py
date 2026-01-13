import copy
import os
import sys

import pandas as pd

from detectors.base_detector import AbstractDetector

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import datetime
import numpy as np
from tqdm import tqdm
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        log_dir,
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model: AbstractDetector = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        # maintain the best metric of all epochs
        self.model.to(device)
        self.model.device = device
        self.training = True
        self.best_metric = {}

        # get current time
        self.time_now = time_now
        # create directory path
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def set_train(self):
        self.model.train()
        self.training = True

    def set_eval(self):
        self.model.eval()
        self.training = False

    def train_step(self, data, epoch):
        x, label, clazz, path = data
        x = x.to(device)
        label = label.to(device)

        use_sam = (self.config['optimizer']['additional'] in ['sam', "sam+is-sam"] and
                   self.config['optimizer']['sam']["start_epoch"] <= epoch)
        use_is_sam = (self.config['optimizer']['additional'] in ['is-sam', "sam+is-sam"] and
                      self.config['optimizer']['sam']["start_epoch"] <= epoch)

        x_original = x.clone().detach()
        if use_is_sam:
            x.requires_grad = True
            predictions = self.model(x)
            loss = self.model.get_losses(label, predictions)
            self.optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = x.grad.data
                grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1).view(-1, 1, 1, 1) + 1e-8
                grad_normalized = grad / grad_norm
                x_perturbed = x_original + self.config['optimizer']['is-sam']['rho'] * grad_normalized
                x_perturbed = torch.clamp(x_perturbed, 0, 1)

            x = x_perturbed
            x.requires_grad = False

        if use_sam:
            predictions = self.model(x)
            loss = self.model.get_losses(label, predictions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            predictions = self.model(x)
            loss = self.model.get_losses(label, predictions)
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            predictions = self.model(x)
            loss = self.model.get_losses(label, predictions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, predictions

    def save_results(self, folder_name, preds, labels, probs, paths, epoch):
        all_preds = np.array(preds)
        all_labels = np.array(labels)
        all_probs = np.array(probs)

        results_df = pd.DataFrame({
            'path': paths,
            'true_label': all_labels,
            'pred_label': all_preds,
            'pred_prob_0': 1 - all_probs,
            'pred_prob_1': all_probs,
            'correct': (all_preds == all_labels).astype(int)
        })

        results_df['true_label_name'] = results_df['true_label'].apply(lambda x: 'real' if x == 0 else 'fake')
        results_df['pred_label_name'] = results_df['pred_label'].apply(lambda x: 'real' if x == 0 else 'fake')

        if not os.path.exists(os.path.join(self.log_dir, folder_name)):
            os.makedirs(os.path.join(self.log_dir, folder_name), exist_ok=True)
        results_csv_path = os.path.join(self.log_dir, folder_name, f"epoch_{epoch}.csv")
        results_df.to_csv(results_csv_path, index=False, encoding='utf-8')
        self.logger.info(f"所有结果已保存到: {results_csv_path}")

    def train_epoch(
            self,
            epoch,
            train_data_loader,
            val_data_loaders=None,
    ):

        self.logger.info("===> Epoch[{}] start, lr= {:.4e}".format(epoch, self.optimizer.param_groups[0]['lr']))
        step_cnt = epoch * len(train_data_loader)

        total_loss = 0
        total_correct = 0
        total_samples = 0

        train_pbar = tqdm(enumerate(train_data_loader), desc=f"Training epoch {epoch}", leave=False,
                          total=len(train_data_loader))
        self.set_train()
        if epoch<=self.config.get("backbone_freeze_epoch", 0):
            for p in self.model.feature_params():
                p.requires_grad = False
        else:
            for p in self.model.feature_params():
                p.requires_grad = True

        all_preds = []
        all_labels = []
        all_probs = []
        all_paths = []

        for iteration, data in train_pbar:
            x, label, clazz, path = data

            loss, logits = self.train_step(data, epoch)

            _, predicted = torch.max(logits, 1)
            correct = (predicted.to(device) == label.to(device)).sum().item()
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            total_correct += correct
            total_samples += label.size(0)
            total_loss += loss.detach().cpu().mean().item()

            batch_acc = correct / label.size(0)
            train_pbar.set_postfix({
                'loss': f'{loss.detach().cpu().mean():.4f}',
                'acc': f'{batch_acc:.4f}'
            })

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(prob.detach()[:, 1].cpu().numpy())
            all_paths.extend(path)

            step_cnt += 1

        avg_loss = total_loss / len(train_data_loader)
        avg_acc = total_correct / total_samples

        self.logger.info(f"===> Epoch[{epoch}] Train - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        self.save_results("train_output", all_preds, all_labels, all_probs, all_paths, epoch)

        self.logger.info("===> Test start!")

        metric = self.val_epoch(epoch, val_data_loaders)

        return metric

    def val_epoch(self, epoch, data_loader):
        self.set_eval()

        all_preds = []
        all_labels = []
        all_probs = []
        all_paths = []
        loss_list = []

        for i, data in tqdm(enumerate(data_loader),total=len(data_loader)):
            x, label, clazz, path = data
            logits = self.inference(data)
            loss = self.model.get_losses(label.to(device), logits.to(device))

            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            loss_list.append(loss.detach().cpu().numpy())

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(prob[:, 1].detach().cpu().numpy())
            all_paths.extend(path)

        self.save_results("val_output", all_preds, all_labels, all_probs, all_paths, epoch)

        metric = {
            "epoch": epoch,
            "avg_loss": np.mean(loss_list),
            "acc": np.mean(np.array(all_preds) == np.array(all_labels)),
            "params": copy.deepcopy(self.model.state_dict())
        }
        self.save_best(metric)
        return metric

    def save_best(self,metric):
        if len(self.best_metric)==0 or self.best_metric["acc"] < metric["acc"]:
            self.best_metric = metric
            torch.save(metric["params"], os.path.join(self.log_dir, "best_model.pth"))
            self.logger.info(f"New best model! Val Acc: {self.best_metric['acc']*100:.2f}%")

    @torch.no_grad()
    def inference(self, data):
        x, label, clazz, path = data
        predictions = self.model(x.to(device), inference=True)
        return predictions
