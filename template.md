# 深度伪造检测测试报告

## 测试概览
- **测试时间**: {timestamp}
- **总样本数**: {total_samples}
- **真实样本数**: {real_samples}
- **伪造样本数**: {fake_samples}
\
## 性能指标
| 指标 | 值 | 描述 |
|------|-----|------|
| 准确率 (ACC) | {accuracy:.4f} | 正确分类的样本比例 |
| 平衡准确率 | {balanced_accuracy:.4f} | 对每个类别准确率的平均 |
| AUC | {auc:.4f} | ROC曲线下面积 |
| AP (Average Precision) | {ap:.4f} | 精确率-召回率曲线下面积 |
| F1 Score | {f1:.4f} | 精确率和召回率的调和平均 |
| 精确率 (Precision) | {precision:.4f} | 预测为伪造的样本中实际为伪造的比例 |
| 召回率 (Recall) | {recall:.4f} | 实际为伪造的样本中被正确预测的比例 |
| EER (Equal Error Rate) | {eer:.4f} | 等错误率 |
| TPR (真正率) | {tpr:.4f} | 实际为正的样本中被正确预测的比例 |
| FPR (假正率) | {fpr:.4f} | 实际为负的样本中被错误预测为正的比例 |
| 对数损失 (Log Loss) | {log_loss:.4f} | 预测概率的对数损失 |

## 混淆矩阵

![混淆矩阵]({confusion_matrix_path})

### 详细统计
- **真实样本准确率**: {real_accuracy:.4f} ({tn}/{tn_fp})
- **伪造样本准确率**: {fake_accuracy:.4f} ({tp}/{tp_fn})

## ROC曲线
![ROC曲线]({roc_curve_path})

## 阈值分析
- **最佳阈值 (Youden's J)**: {best_threshold:.4f}
- **对应的TPR**: {best_tpr:.4f}
- **对应的FPR**: {best_fpr:.4f}
