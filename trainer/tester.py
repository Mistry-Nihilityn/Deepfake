import json
import os
import sys
from datetime import datetime

import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, \
    roc_curve, average_precision_score, log_loss, confusion_matrix, ConfusionMatrixDisplay

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import numpy as np
from tqdm import tqdm
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tester(object):
    def __init__(
        self,
        config,
        model,
        logger,
        log_dir,
        ):
        self.config = config
        self.model = model
        self.logger = logger
        self.model.to(device)
        self.model.device = device
        self.best_metric = {}
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _plot_roc_curve(self, fpr, tpr, auc_score, save_path):
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC Curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)

        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = (fpr[eer_idx] + 1 - tpr[eer_idx]) / 2
        plt.scatter(fpr[eer_idx], tpr[eer_idx], color='red', s=100,
                    label=f'EER = {eer:.4f}', zorder=5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')

        plt.text(0.6, 0.3, f'AUC = {auc_score:.4f}\nEER = {eer:.4f}',
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return eer

    def _plot_confusion_matrix(self, all_labels, all_preds, save_path):
        cm = confusion_matrix(all_labels, all_preds)
        class_names = ['Fake', 'Real']

        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.title('Confusion Matrix - Test Set\n', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm[i].sum() * 100
                ax.text(j, i-0.2, f'({percentage:.1f}%)',
                        ha='center', va='center', color='darkred', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_best_threshold(self, fpr, tpr, thresholds):
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        return thresholds[best_idx], tpr[best_idx], fpr[best_idx]

    def _calculate_eer(self, fpr, tpr):
        frr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - frr))
        eer = (fpr[eer_idx] + frr[eer_idx]) / 2
        return eer

    def test(self, test_data_loader):
        self.logger.info("===> Test start!")
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        start_time = datetime.now()

        with torch.no_grad():
            train_pbar = tqdm(enumerate(test_data_loader), desc=f"Testing",
                              leave=False, total=len(test_data_loader))
            for iteration, data in train_pbar:
                x, label = data
                label = label.to(device)

                logits = self.inference(data)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        test_duration = datetime.now() - start_time

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {}
        accuracy = accuracy_score(all_labels, all_preds)
        metrics['ACC'] = accuracy.item()

        try:
            auc_score = roc_auc_score(all_labels, all_probs)
            metrics['AUC'] = auc_score.item()
        except:
            auc_score = 0.0
            metrics['AUC'] = 0.0

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

        f1 = f1_score(all_labels, all_preds, average='binary')
        metrics['F1'] = f1.item()

        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        metrics['Precision'] = precision.item()
        metrics['Recall'] = recall.item()

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        metrics['TP'] = int(tp.item())
        metrics['FP'] = int(fp.item())
        metrics['FN'] = int(fn.item())
        metrics['TN'] = int(tn.item())

        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['TPR'] = tpr_val.item()
        metrics['FPR'] = fpr_val.item()

        # 计算平衡准确率
        real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        balanced_acc = (real_acc + fake_acc) / 2
        metrics['Balanced_ACC'] = balanced_acc.item()

        ap = average_precision_score(all_labels, all_probs)
        metrics['AP'] = ap.item()

        labels_one_hot = np.eye(2)[all_labels.astype(int)]
        probs_matrix = np.column_stack([1 - all_probs, all_probs])
        log_loss_value = log_loss(labels_one_hot, probs_matrix)
        metrics['Log_Loss'] = log_loss_value.item()

        eer = self._calculate_eer(fpr, tpr)
        metrics['EER'] = eer.item()

        best_threshold, best_tpr, best_fpr = self._calculate_best_threshold(fpr, tpr, thresholds)
        metrics['Best_Threshold'] = best_threshold.item()
        metrics['Best_TPR'] = best_tpr.item()
        metrics['Best_FPR'] = best_fpr.item()

        roc_curve_path = os.path.join(self.log_dir, "roc_curve.png")
        confusion_matrix_path = os.path.join(self.log_dir, "confusion_matrix.png")
        self._plot_roc_curve(fpr, tpr, auc_score, roc_curve_path)
        self._plot_confusion_matrix(all_labels, all_preds, confusion_matrix_path)

        self._generate_report({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': len(all_labels),
            'real_samples': np.sum(all_labels == 0),
            'fake_samples': np.sum(all_labels == 1),
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'auc': auc_score,
            'ap': ap,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'eer': eer,
            'tpr': tpr_val,
            'fpr': fpr_val,
            'log_loss': log_loss_value,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn_fp': tn + fp,
            'tp_fn': tp + fn,
            'real_accuracy': real_acc,
            'fake_accuracy': fake_acc,
            'roc_curve_path': "roc_curve.png",
            'best_threshold': best_threshold,
            'best_tpr': best_tpr,
            'best_fpr': best_fpr,
            'confusion_matrix_path': "confusion_matrix.png"
        })

        self._save_metrics(metrics)
        self.logger.info("Metrics: {}".format(metrics))
        return metrics

    @torch.no_grad()
    def inference(self, data):
        x, label = data
        predictions = self.model(x.to(device), inference=True)
        return predictions

    def _generate_report(self, report_data):
        with open(os.path.join(parent_dir, "template.md"), 'r', encoding='utf-8') as f:
            report_template = f.read()
        report_content = report_template.format(**report_data)

        report_path = os.path.join(self.log_dir, "test_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"Report saved at {report_path}")

    def _save_metrics(self, metrics):
        json_path = os.path.join(self.log_dir, "test_metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        self.logger.info(f"Metrics saved at {json_path}")

    def _save_config(self, config):
        yaml_path = os.path.join(self.log_dir, "configs.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config)
        self.logger.info(f"Config saved at {yaml_path}")