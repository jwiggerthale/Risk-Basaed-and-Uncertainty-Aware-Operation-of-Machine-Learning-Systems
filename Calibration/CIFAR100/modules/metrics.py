'''
This script implements different evaluation metrics applied during benchmarking of UQ techiques in Chapter 5 of my PhD Thesis
'''



from __future__ import annotations
from scipy.interpolate import interp1d
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import pandas as pd
from typing import Sequence, Callable
from math import erf, sqrt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Common metrics

def accuracy(preds: list, 
             labels: list):
    preds = np.array(preds)
    labels = np.array(labels)
    pred_cls = preds.argmax(axis = 1)
    acc = (pred_cls == labels).sum()
    acc /= len(labels)
    return acc


def get_f1_score(preds: list, 
                labels: list):
    preds = np.array(preds)
    labels = np.array(labels)
    pred_cls = preds.argmax(axis = 1)
    f1 = f1_score(pred_cls, labels, average='weighted')
    return f1


# ece
'''
call with:
  samples: np.array --> predictions of models (sgape [num_samples, num_classes])
  true_labels: np.array --> true labels
  m: int --> number of bins
returns: 
  ece: float --> ece
'''
def expected_calibration_error(samples, true_labels, M=20):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()
        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece[0]



# different functions to calculate skce
# most of them are called automatically

# one hot encoding for true labels
def one_hot(labels: np.ndarray, m: int) -> np.ndarray:
    labels = labels.astype(int)
    oh = np.zeros((labels.shape[0], m), dtype=float)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh


# estimator for distance between predictions P
# samples pairs from P and calculates euclidean distance 
# median of euclidean distance is returned
"""
function calculates nu via median heuritic: median(||p_i - p_j||) over random pairs
use when n is large
"""
def median_heuristic_bandwidth(P: np.ndarray, 
                               max_pairs: int = 200_000, 
                               rng: int = 0):
    n = P.shape[0]
    if n < 2:
        return 1.0
    rs = np.random.default_rng(rng)
    # sample random pairs (i<j) without building full distance matrix
    k = min(max_pairs, n * (n - 1) // 2)
    i = rs.integers(0, n, size=k)
    j = rs.integers(0, n, size=k)
    mask = i != j
    i, j = i[mask], j[mask]
    d = np.linalg.norm(P[i] - P[j], axis=1)  # Euclidean norm
    med = np.median(d)
    return float(med if med > 0 else 1e-12)



'''
kernel to compute pairwise euclidean distance between target distribution Q and predicted distribution P
scales by nu (median euclidean distance within P)
Scalar kernel  k~(p,q) = exp(-||p-q||/nu).
Returns shape (len(P), len(Q)).
'''
def laplacian_kernel_scalar(P: np.ndarray, 
                            Q: np.ndarray, 
                            nu: float) -> np.ndarray:
    # compute pairwise Euclidean distances in a vectorized way:
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    P2 = np.sum(P * P, axis=1, keepdims=True)      # (n,1)
    Q2 = np.sum(Q * Q, axis=1, keepdims=True).T    # (1,m)
    sq = np.maximum(P2 + Q2 - 2.0 * (P @ Q.T), 0.0)
    dist = np.sqrt(sq)
    return np.exp(-dist / max(nu, 1e-12))



# estimator for skce with laplacian kernel 
# returns skce and kce
# methods b, uq and ul from widmann et al. are implemented
def skce_estimators_scalarI(
    probs: np.ndarray,          # shape (samples, classes), rows sum to 1
    labels: np.ndarray,         # shape (samples), int in [0, classes-1]
    nu: float | None = None,
    estimator: str = "uq",      # "b" (biased), "uq" (unbiased, quadratic runtime), "ul" (unbiased, linear runtime) 
    block_size: int | None = None,
    rng: int = 0,
) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n, m = probs.shape
    assert labels.shape == (n,)
    # normalize to make sure rows sum to 1
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, row_sums, out=np.zeros_like(probs), where=row_sums != 0)
    # get nu 
    if nu is None:
        nu = median_heuristic_bandwidth(probs, rng=rng)  # used in paper experiments (widmann et al.
    R = one_hot(labels, m) - probs  # residuals r_i = e_{y_i} - p_i
    est = estimator.lower()
    if est == "ul":
        # linear-time unbiased estimator: average over pairs (1,2), (3,4),
        t = n // 2
        if t == 0:
            skce = 0.0
        else:
            i = np.arange(0, 2 * t, 2)
            j = i + 1
            k_ij = np.exp(-np.linalg.norm(probs[i] - probs[j], axis=1) / max(nu, 1e-12)) # kernel function
            dot = np.sum(R[i] * R[j], axis=1)
            skce = float(np.mean(k_ij * dot))
        return {"SKCE": skce, "KCE": float(np.sqrt(max(skce, 0.0))), "nu": float(nu)}
    # quadratic estimators: can be O(n^2) time; memory depends on block_size
    if block_size is None:
        # full matrix version (fast but O(n^2) memory)
        K = laplacian_kernel_scalar(probs, probs, nu=nu)  # (n,n)
        G = R @ R.T                                      # (n,n)  dot products
        H = K * G                                        # (n,n)  h_ij values for scalar*I kernel
        if est == "b":
            skce = float(np.mean(H))  # (1/n^2) sum_{i,j} h_ij 
        elif est == "uq":
            s = np.triu(H, k=1).sum()
            skce = float(2.0 * s / (n * (n - 1)))  # 2/(n(n-1)) sum_{i<j} h_ij
        else:
            raise ValueError("estimator must be 'b', 'uq', or 'ul'")
        return {"SKCE": skce, "KCE": float(np.sqrt(max(skce, 0.0))), "nu": float(nu)}
    # blockwise version: O(n^2) time, O(block_size^2) memory
    B = int(block_size)
    if B <= 0:
        raise ValueError("block_size must be positive")
    total = 0.0
    count = 0
    if est == "b":
        # sum over all (i,j)
        for a0 in range(0, n, B):
            A = slice(a0, min(a0 + B, n))
            PA, RA = probs[A], R[A]
            for b0 in range(0, n, B):
                Bsl = slice(b0, min(b0 + B, n))
                PB, RB = probs[Bsl], R[Bsl]
                K = laplacian_kernel_scalar(PA, PB, nu=nu)
                G = RA @ RB.T
                total += float(np.sum(K * G))
                count += (PA.shape[0] * PB.shape[0])
        skce = total / max(count, 1)
    elif est == "uq":
        # sum over i<j only
        for a0 in range(0, n, B):
            A = slice(a0, min(a0 + B, n))
            PA, RA = probs[A], R[A]
            # diagonal block: take upper triangle only
            Kaa = laplacian_kernel_scalar(PA, PA, nu=nu)
            Gaa = RA @ RA.T
            Haa = Kaa * Gaa
            total += float(np.triu(Haa, k=1).sum())
            # off-diagonal blocks: a < b, take all entries
            for b0 in range(a0 + B, n, B):
                Bsl = slice(b0, min(b0 + B, n))
                PB, RB = probs[Bsl], R[Bsl]
                Kab = laplacian_kernel_scalar(PA, PB, nu=nu)
                Gab = RA @ RB.T
                total += float(np.sum(Kab * Gab))
        skce = 2.0 * total / (n * (n - 1)) if n > 1 else 0.0

    else:
        raise ValueError("estimator must be 'b', 'uq', or 'ul'")

    return {"SKCE": float(skce), "KCE": float(np.sqrt(max(skce, 0.0))), "nu": float(nu)}





def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

'''
function to calculate p value for skce as proposed by widmann et al. 
returns dict with p SKCE, p_value and nu
'''
def p_value_skce_ul(probs: np.ndarray, labels: np.ndarray, nu: float | None = None):
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    n, m = probs.shape
    if nu is None:
        nu = median_heuristic_bandwidth(probs)

    R = one_hot(labels, m) - probs
    t = n // 2
    if t == 0:
        return {"SKCE_ul": 0.0, "p_value": 1.0, "nu": float(nu)}

    i = np.arange(0, 2 * t, 2)
    j = i + 1
    k_ij = np.exp(-np.linalg.norm(probs[i] - probs[j], axis=1) / max(nu, 1e-12))
    h = k_ij * np.sum(R[i] * R[j], axis=1)  # the paired h_{2i-1,2i}
    skce_ul = float(np.mean(h))
    s_hat = float(np.std(h, ddof=1)) if t > 1 else 0.0

    # Lemma 3 implies sqrt(t) * SKCE_ul approx Normal(0, s_hat) under H0
    if s_hat <= 0:
        p = 1.0 if skce_ul <= 0 else 0.0
    else:
        z = sqrt(t) * skce_ul / s_hat
        p = 1.0 - normal_cdf(z)  # one-sided: large SKCE => reject calibration
    return {"SKCE_ul": skce_ul, "p_value": float(p), "nu": float(nu)}



'''
kernel for skce
'''
def hij_general_matrix_kernel(p_i, y_i, p_j, y_j, k_mat):
    """
    k_mat(p,q) -> (m,m) matrix
    """
    m = p_i.shape[0]
    e_i = np.zeros(m); e_i[y_i] = 1.0
    e_j = np.zeros(m); e_j[y_j] = 1.0
    r_i = e_i - p_i
    r_j = e_j - p_j
    K = k_mat(p_i, p_j)          # (m,m)
    return float(r_i @ K @ r_j)  # (e_yi - p_i)^T k(p_i,p_j) (e_yj - p_j)




# below follow function for plotting



'''
function to plot calibration curve for multi class claissification
call with: 
  probs : array-like, shape (n_samples, n_classes) --> predicted probs
  labels : array-like, shape (n_samples,) --> true labels
  n_bins : int --> bins to calculate ece
  strategy : str --> uniform or quantile 
# note: not used to create plots in thesis
'''
def plot_multiclass_calibration(probs: list, labels:list, 
                                classes: list = [1, 34, 72],
                                n_bins: int =10, 
                                strategy: str ='uniform', 
                                save_path: str = 'calibration_curve_per_class.png'):
    probs = np.array(probs)
    labels = np.array(labels)
    n_classes = probs.shape[1]
    # subplot per class + aggregated 
    fig, axes = plt.subplots(2, len(classes) + 1 // 2, figsize=(15, 10))
    axes = axes.flatten()
    # plot per class
    for i, class_idx in enumerate(classes):
        ax = axes[i]
        # Konvertiere zu binärem Problem: Klasse vs. Rest
        y_binary = (labels == class_idx).astype(int)
        y_prob = probs[:, class_idx]
        # Berechne Calibration Curve
        try:
            prob_true, prob_pred = calibration_curve(
                y_binary, y_prob, n_bins=n_bins, strategy=strategy
            )
            
            # calibration curve
            ax.plot(prob_pred, prob_true, marker='o', label=f'Class {class_idx}')
        except ValueError as e:
            # if insufficient number of samples in class
            ax.text(0.5, 0.5, f'Not enough values\nfor class {class_idx}',
                   ha='center', va='center', transform=ax.transAxes)
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Prefect calibration') 
        ax.set_xlabel('Mean probability')
        ax.set_ylabel('Share of positive samples')
        ax.set_title(f'Calibration Curve - Class {class_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])  
    plt.tight_layout()
    plt.savefig(save_path)


'''
Function to plot aggregated calibration curve
'''
def plot_aggregated_calibration_curve(
    probs: np.array, 
    labels: np.array,
    n_bins: int = 15,
    *,
    title: str = "Aggregated Calibration Curve (Top-1)", 
    save_path: str = 'Calibration_Curve.png'
):
    probs = np.array(probs)
    labels = np.array(labels)
    pred_cls = probs.argmax(axis = 1)
    probs = probs.max(axis = 1)
    correct = (pred_cls  == labels).astype(np.float64)
    n = probs.shape[0]

    # Binning
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bin_edges[1:-1], right=False)  # 0..n_bins-1

    bin_acc = np.full(n_bins, np.nan, dtype=np.float64)
    bin_conf = np.full(n_bins, np.nan, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        m = (bin_ids == b)
        bin_count[b] = int(m.sum())
        if bin_count[b] > 0:
            bin_acc[b] = correct[m].mean()
            bin_conf[b] = probs[m].mean()

    # ECE (weighted avg |acc - conf|)
    gaps = np.abs(bin_acc - bin_conf)
    gaps[np.isnan(gaps)] = 0.0
    ece = float(np.sum((bin_count / max(n, 1)) * gaps))

    acc_total = float(correct.mean()) if n > 0 else float("nan")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    # Nur belegte Bins plotten
    valid = bin_count > 0
    x_vals = bin_conf[valid]
    y_vals = bin_acc[valid]

    # Reliability diagram: Punkte/Line + Diagonale
    ax.plot([0, 1], [0, 1], linestyle="--")  # perfect calibration
    #ax.plot(x_vals, y_vals, marker="o")
    if len(x_vals) > 3:  # Need at least 4 points for cubic
        print('applying smooothing')
        f = interp1d(x_vals, y_vals, kind='cubic', bounds_error=False, fill_value='extrapolate')
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
        y_smooth = f(x_smooth)
        ax.plot(x_smooth, y_smooth, marker="", linewidth=2)
    else:
        ax.plot(x_vals, y_vals, marker="o")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)

    #  small info box
    text = f"N={n}\nAcc={acc_total:.4f}\nECE={ece:.4f}\nBins={n_bins}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top")

    metrics = {
        "n_samples": float(n),
        "accuracy": acc_total,
        "ece": ece,
    }
    plt.savefig(save_path)


'''
Another function(more simple) to plot aggregated calibration curve
treats each prediction as binary event (correct or wrong)
'''
def plot_aggregated_calibration(probs, 
                                labels, 
                                n_bins=10, 
                                strategy='uniform', 
                                save_path: str = 'aggregated_calibration_curve.png'):
    pred_classes = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)
    # Binary outcome: Vorhersage korrekt?
    correct = (pred_classes == labels).astype(int)
    prob_true, prob_pred = calibration_curve(
        correct, pred_probs, n_bins=n_bins, strategy=strategy
    )
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, 
            label='Model Calibration')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Calibration Curve (Top-1)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path)




'''
Function to plot calibration curve (binary)
'''
def plot_calibration_curve(y_true: list, 
                           probs:list, 
                           bins: int = 20, 
                           strategy: str = 'uniform', 
                           save_path: str = './visualizations/calibration_curve.png'):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, probs, n_bins=bins, strategy=strategy)
    max_val = max(mean_predicted_value)
    plt.figure(figsize=(8,10))
    plt.subplot(2, 1, 1)
    plt.plot(mean_predicted_value, fraction_of_positives, label='Model') 
    plt.plot(
        np.linspace(0, max_val, bins), 
        np.linspace(0, max_val, bins), 
        linestyle='--', 
        color='red', 
        label='Perfect calibration'
    ) 
    plt.xlabel('Probability Predictions') 
    plt.ylabel('Empirical Accuracy')
    plt.title('Calibration Curve') 
    plt.legend(loc='upper left') 
    plt.subplot(2, 1, 2) 
    plt.hist(probs, range=(0, 1), bins=bins, density=True, stacked=True, alpha=0.3) 
    plt.savefig(save_path)
    plt.show()




# Abstained prediction test
'''
Function which evaluates models predictions on a metric function
Called automatically from function evaluate 
Call with: 
  y_true --> labels
  y_pred --> predictions
  y_uncertainty --> uncertainties
  fractions --> list of fractions to be retained
  metric_fn --> sklearn.metrics.accuracy_score or sklearn.metrics.roc_auc_score
Returns: 
  DataFrames for performance
    --> each DataFrame has column mean and fraction
    --> mean is AUC / accuracy of model for appropriiate fraction
'''
def abstained_prediction(
      y_true: list,
      probs: list,
      y_uncertainty: list,
      fractions: Sequence[float],
    ) -> pd.DataFrame:
    y_true = np.array(y_true)
    y_uncertainty = np.array(y_uncertainty)
    y_pred = np.array(probs)
    N = y_true.shape[0]
    
    # Sorts indexes by ascending uncertainty
    I_uncertainties = np.argsort(y_uncertainty)
    
    # Score containers
    mean = np.empty_like(fractions)
    # TODO(filangel): do bootstrap sampling and estimate standard error
    std = np.zeros_like(fractions)
    
    for i, frac in enumerate(fractions):
      # Keep only the %-frac of lowest uncertainties
      I = np.zeros(N, dtype=bool)
      I[I_uncertainties[:int(N * frac)]] = True
      I = np.array(I)
      mean[i] = accuracy(labels = y_true[I], preds = y_pred[I])
    
    # Store
    df = pd.DataFrame(dict(retained_data=fractions, mean=mean, std=std))
    
    return df


'''
Function to plot abstained prediction curve
'''
def plot_abstained_prediction(results: dict, 
                              save_path: str = './visualizations/abstained_prediction_curve.png'):
    
    fig, ax = plt.subplots(figsize =(12,12))
    for key, value in results.items():
        ax.plot(value['retained_data'], value['mean'], label = key)
    plt.title('Accuracy Development in Abstained Prediction Test', fontsize = 22)
    plt.xlabel('Share of Samples Classified', fontsize = 22)
    plt.ylabel('Accuracy', fontsize = 22)
    plt.savefig(save_path)




"""
Function to compute AUPR for an uncertainty score u w.r.t. a binary target y.
Call with:
  u : array-like, shape (n,) --> Uncertainty scores. Larger should mean "more positive" (e.g., more likely error/OOD).
  y : array-like, shape (n,) --> Binary labels (0/1). 1 = positive class (e.g., error).
Returns:
    AUPR --> float in [0, 1]

Note: not used for thesis
"""
def aupr_from_uncertainty(unc: np.ndarray, # model uncertainty
                          is_wrong: np.ndarray # 1 if prediction is wrong and 0 if prediction is correct 
                          ) -> float:
    precision, recall, thresholds = precision_recall_curve(is_wrong, unc)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall, precision)
    return auc_precision_recall



'''
Function to calculate aurc
call with:
    y_true: array-like, shape (N,)
    y_pred: array-like, shape (N,)
    uncertainty: array-like, shape (N,) --> higher = more uncertain
'''
def compute_aurc(y_true, 
                 y_pred, 
                 uncertainty):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    uncertainty = np.asarray(uncertainty)
    # 1) errors
    errors = (y_pred != y_true).astype(float)
    # 2) sort by uncertainty
    order = np.argsort(uncertainty)
    errors_sorted = errors[order]
    # 3) cumulated error
    cum_errors = np.cumsum(errors_sorted)
    # 4) coverage and risk
    k = np.arange(1, len(errors_sorted) + 1)
    coverage = k / len(errors_sorted)
    risk = cum_errors / k
    # 5) numerical integration
    aurc = np.trapz(risk, coverage)
    return aurc, coverage, risk



'''
Function to calculate NLL
call with:
    y_true: array-like, shape (N,)
    probs: array-like, shape (N, C), rows sum to 1
'''
def compute_nll_multiclass(y_true, probs, eps=1e-12):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    probs = np.clip(probs, eps, 1.0)
    # Wahrscheinlichkeit der wahren Klasse
    p_true = probs[np.arange(len(y_true)), y_true]
    nll = -np.mean(np.log(p_true))
    return nll
