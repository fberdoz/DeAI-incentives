from matplotlib import pyplot as plt
from helpers import evaluate_model
import numpy as np


def perfplots(perf_dic, metrics='all', series='all', legends=None, suptitle=None, log_loss=True):
    """Plot the performance history of the training phase."""
    
    # 'metrics' argument processing
    if type(metrics) is not list and metrics is not 'all':
        metrics = [metrics]
    elif metrics is 'all':
        metrics = list(perf_dic.keys())

    # 'series' argument processing
    if perf_dic[metrics[0]].ndim != 2:
        raise ValueError('Data must be 2D.')
    n_series = perf_dic[metrics[0]].shape[1]
    if type(series) is int:
        series = [series]
    elif series is 'all':
        series = range(n_series)
        
    if max(series)>= n_series or min(series) < 0:
        raise ValueError("Invalid argument 'series'.")
    
    # 'legends' argument processing
    if legends is not None:
        if all([type(leg)==float for leg in legends]) and len(legends) == len(series):
            legends = ["{}%".format(100 * leg) for leg in legends]
        elif not all([type(leg)==str for leg in legends]) or len(legends) != len(series):
            raise ValueError("Invalid 'legends' argument.")   
            
    # Plots
    nrow = len(metrics)
    fig, axs = plt.subplots(nrow ,1, figsize=(6, nrow*2), sharex=True)
    fig.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    for idx, metric in enumerate(metrics):
        axs[idx].grid()
        axs[idx].set_ylabel(metric)
        if idx == nrow-1:
            axs[idx].set_xlabel('round')
            
        if metric is not 'loss_norm':
            axs[idx].set_ylim(0, 1)
        elif metric is 'loss_norm' and log_loss:
            axs[idx].set_yscale("log")
        for serie in series:
            axs[idx].plot(perf_dic[metric][:, serie])

                
        if legends is not None:
            if metric is 'loss_norm':
                axs[idx].legend(legends, loc="upper right")
            else:
                axs[idx].legend(legends, loc="lower right")
    return fig

def ROC(model, loader, npoints=101, annotate=[5, 10, 20, 30, 50, 70, 90]):
    """Create a ROC plot for a binary classification model."""
    thresholds = np.linspace(0, 1, npoints)
    
    TPR = np.empty_like(thresholds)
    FPR = np.empty_like(thresholds)
    
    for i, t in enumerate(thresholds):
        print('Building the ROC plot ({}/{} evaluations)'.format(i+1, npoints), end='\r')
        perf = evaluate_model(model, loader, threshold=t)
        TPR[i] = perf['recall']
        FPR[i] = perf['FPR']

    fig, ax = plt.subplots(1 ,1, figsize=(4, 4))
    fig.suptitle('ROC Plot')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.grid()
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.plot(thresholds, thresholds, 'r--')
    if annotate is not None:
        ax.plot(FPR[annotate], TPR[annotate], 'k.')
        for i in annotate:
            if i < 11:
                ax.annotate('{:.2f}'.format(thresholds[i]), (FPR[i], TPR[i]), xytext=(5,-10), textcoords='offset points')
            else:
                ax.annotate('{:.1f}'.format(thresholds[i]), (FPR[i], TPR[i]), xytext=(5,-10), textcoords='offset points')
    
    ax.plot(FPR, TPR)
    
    return fig
    
def contriplot(contributions, legends=None, normalize=False, suptitle='Contribution Plot', 
               modes=['round', 'round_maxmax', 'round_bar', 'cum', 'pos', 'neg', 'minmax', 'maxmax']):
    
    # parameter extraction
    num_rounds, num_clients = contributions.shape
    cum_contributions = np.cumsum(contributions, axis=0)
    fontsize=10
    
    # 'legends' argument processing
    if legends is not None:
        if all([type(leg)==float for leg in legends]) and len(legends) == num_clients:
            legends = ["{}%".format(100 * leg) for leg in legends]
        elif not all([type(leg)==str for leg in legends]) or len(legends) != num_clients:
            raise ValueError("Invalid 'legends' argument.")   
    
    # figure creation
    n_rows = len(modes)
    fig, axs = plt.subplots(n_rows , 1, figsize=(6, n_rows * 1.6), sharex=True, edgecolor='black', linewidth=0)
    
    fig.suptitle(suptitle, fontsize=fontsize)
    #plt.tight_layout(rect=[0, 0.2, 1, 0.9])
    x = np.arange(num_rounds)
    
    for i, mode in enumerate(modes):
        axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[i].grid()
        if i==n_rows-1:
            axs[i].set_xlabel('Round')
        
        if mode == 'round':
            axs[i].set_title('Per round', loc='right', fontsize=fontsize)
            for j in range(num_clients):
                axs[i].plot(x, contributions[:, j])
            
            if legends is not None:
                axs[i].legend(legends, loc="upper left")       
        
        elif mode == 'round_bar':
            delta = 1/(num_clients + 1)
            axs[i].set_title('Per round', loc='right', fontsize=fontsize)
            for j in range(num_clients):
                
                axs[i].bar(x + j*delta, contributions[:, j], width=delta)
            if legends is not None:
                axs[i].legend(legends, loc="upper left")
                
        elif mode == 'cum':
            axs[i].set_title('Cumulative', loc='right', fontsize=fontsize)
            
            for j in range(num_clients):
                axs[i].plot(x, cum_contributions[:, j])
                
            if legends is not None:
                axs[i].legend(legends, loc="upper left")
                
        elif mode == 'pos':
            pos_score = np.cumsum(np.where(contributions > 0, contributions, 0), axis=0)

            div_pos=1
            if normalize:
                div_pos = pos_score.sum(axis=1)
                div_pos[div_pos <= 1e-8] = 1e-8
                axs[i].set_ylim(0, 1)
                
            axs[i].set_title('Positive Contribution', loc='right', fontsize=fontsize)
            
            for j in range(num_clients):
                axs[i].plot(x, pos_score[:, j] / div_pos)

            if legends is not None:
                axs[i].legend(legends, loc="upper left")
                
        elif mode == 'neg':
            neg_score =  np.cumsum(np.where(contributions < 0, -contributions, 0), axis=0)
            div_neg=1
            if normalize:
                div_neg = neg_score.sum(axis=1)
                div_neg[div_pos <= 1e-8] = 1e-8
                axs[i].set_ylim(0, 1)
            axs[i].set_title('Negative Contribution', loc='right', fontsize=fontsize)
            
            for j in range(num_clients):
                axs[i].plot(x, neg_score[:, j] / div_neg)
                
            if legends is not None:
                axs[i].legend(legends, loc="upper left")
        
        elif mode == 'minmax':
            minmax_contri = (cum_contributions - cum_contributions.min(axis=1, keepdims=True)) / ...
            (cum_contributions.max(axis=1, keepdims=True) - cum_contributions.min(axis=1, keepdims=True)) 
            
            axs[i].set_title('Min-Max Normalization (Cumulative)', loc='right', fontsize=fontsize)
            
            for j in range(num_clients):
                axs[i].plot(x, minmax_contri[:, j])
                
            if legends is not None:
                axs[i].legend(legends, loc="upper left")       
        
        elif mode == 'round_minmax':
            minmax_contri_r = (contributions - contributions.min(axis=1, keepdims=True)) / ...
            (contributions.max(axis=1, keepdims=True) - contributions.min(axis=1, keepdims=True)) 
            
            axs[i].set_title('Min-Max Normalization', loc='right', fontsize=fontsize)
            
            for j in range(num_clients):
                axs[i].plot(x, minmax_contri_r[:, j])
                
            if legends is not None:
                axs[i].legend(legends, loc="upper left")       
        
        elif mode == 'maxmax':
            maxmax_contri = (cum_contributions + cum_contributions.max(axis=1, keepdims=True)) / ...
            (2*cum_contributions.max(axis=1, keepdims=True)) 
            
            axs[i].set_title('Max-Max Normalization (Cumulative)', loc='right', fontsize=fontsize)
            
            for j in range(num_clients):
                axs[i].plot(x, maxmax_contri[:, j])
                
            if legends is not None:
                axs[i].legend(legends, loc="upper left")   
        
        elif mode == 'round_maxmax':
            maxmax_contri_r = (contributions + contributions.max(axis=1, keepdims=True)) / ...
            (2*contributions.max(axis=1, keepdims=True)) 
            
            axs[i].set_title('Max-Max Normalization', loc='right', fontsize=fontsize)
            
            for j in range(num_clients):
                axs[i].plot(x, maxmax_contri_r[:, j])
                
            if legends is not None:
                axs[i].legend(legends, loc="upper left")   
        else:
            raise ValueError('Unknown mode')
            
    
    return fig