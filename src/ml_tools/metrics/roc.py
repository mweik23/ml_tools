from sklearn.metrics import roc_curve
import numpy as np
from matplotlib import pyplot as plt

def buildROC(labels, score, targetEff=[0.3,0.5]):
    r''' ROC curve is a plot of the true positive rate (Sensitivity) in the function of the false positive rate
    (100-Specificity) for different cut-off points of a parameter. Each point on the ROC curve represents a
    sensitivity/specificity pair corresponding to a particular decision threshold. The Area Under the ROC
    curve (AUC) is a measure of how well a parameter can distinguish between two diagnostic groups.
    '''
    if not isinstance(targetEff, list):
        targetEff = [targetEff]
    fpr, tpr, threshold = roc_curve(labels, score)
    idx = [np.argmin(np.abs(tpr - Eff)) for Eff in targetEff]
    eB, eS = fpr[idx], tpr[idx]
    return fpr, tpr, threshold, eB, eS
def make_roc_curve(fpr, tpr, domain='Source', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    mask = fpr>0
    ax.plot(tpr[mask], 1/(fpr[mask]), label=domain)
    return ax

def get_test_metrics(labels, scores, targetEff=[0.3, 0.5], **kwargs):
    fpr, tpr, _, eB, eS = buildROC(labels, scores, targetEff=targetEff)
    auc = np.trapz(tpr, fpr)
    ax = make_roc_curve(fpr, tpr, **kwargs)
    metrics = {f'1/eB ~ {eff}': 1/eB[i] for i, eff in enumerate(targetEff)}
    metrics['auc'] = auc
    return metrics, ax