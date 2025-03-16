import numpy as np
import shap
from typing import List
from tqdm.auto import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
# sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

def cv_splits(
    years, 
    cv_splits,
    dataset='env', 
    save_name=None,
):

    # book keeping
    train_idxs, val_idxs, test_idx = cv_splits

    # initialize plot
    plt.figure(figsize=(20, 0.5*(len(train_idxs)+1)+1))
    if dataset == 'env':
        plt.title('Environmental time series cross validation folds')
    elif dataset == 'bat':
        plt.title('Bat-level time series cross validation folds')

    # plot decorations
    for i in np.arange(years.min(), years.max()+2):
        plt.axvline(i, c='k', linewidth=0.5, alpha=0.5)

    # plot train/val/test ranges
    b, o, g = 'tab:blue', 'tab:orange', 'tab:green'
    for i, (t, v) in enumerate(zip(train_idxs, val_idxs)):
        t_start, t_stop = years[t].min(), years[t].max()+1
        v_start, v_stop = years[v].min(), years[v].max()+1
        fold = i+1
        plt.plot([t_start, t_stop], [fold, fold], '-', c=b, linewidth=20, 
                 label='Training set')
        plt.plot([v_start, v_stop], [fold, fold], '-', c=o, linewidth=20, 
                 label='Validation set')
    tr_start, tr_stop = years.min(), years[test_idx].min()
    te_start, te_stop = years[test_idx].min(), years[test_idx].max()+1
    fold = len(train_idxs) + 1
    plt.plot([tr_start, tr_stop], [fold, fold], '-', c=b, linewidth=20, 
             label='Training set')
    plt.plot([te_start, te_stop], [fold, fold], '-', c=g, linewidth=20, 
             label='Test set')

    # deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    # axis decorations
    plt.ylim([len(train_idxs)+1.5, 0.5])
    plt.xlim([years.min()-0.5, years.max()+0.5])
    plt.yticks(
        np.arange(1, len(train_idxs)+2), 
        ['Fold {}'.format(i+1) for i in range(len(train_idxs))] + ['Test'])
    plt.xticks(
        np.arange(years.min(), years.max()+1)+0.6, 
        np.arange(years.min(), years.max()+1), 
        rotation=90)
    plt.gca().yaxis.grid(False)
    plt.gca().xaxis.grid(False)

    # show
    plt.tight_layout()
    if save_name is not None:
        save_name += '.pdf' if save_name[-4:] != '.pdf' else ''
        plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.show()

def plot_missing(
    df_missing, 
    features, 
    rename, 
    verbose=True,
    save_name=None,
):

    missing = ~df_missing[features].isna().values.T # [features, months]
    f_missing = np.sum(~missing, axis=1)
    p_missing = f_missing / len(df_missing) * 100 # proportions

    y = np.sum(~missing, axis=0)/len(missing)
    m, b = np.polyfit(np.arange(len(y)), y, 1)
    z = np.poly1d([m, b])

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].imshow(~missing, aspect='auto', cmap='gray')
    axs[0].set_title('Heatmap of missing bat-level observations')
    axs[0].set_yticks(ticks=np.arange(len(features)), labels=[rename[f] for f in features])
    axs[0].set_xticks(
        ticks=np.arange(0, len(df_missing), 12),
        labels=np.arange(df_missing['cal_year'].min(), df_missing['cal_year'].max())
    )
    axs[0].legend(
        handles=[
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='k', linewidth=0, markersize=15, label='Present'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='w', linewidth=0, markersize=15, label='Missing'),
        ],
        loc='upper left',
        framealpha=1
    )

    axs[1].plot(y, 'ko', label='Missing')
    axs[1].plot(z(np.arange(len(y))), 'k-', label='Trend')
    axs[1].set_ylim(-0.03, 1.03)
    axs[1].set_yticks(
        ticks=np.arange(0, 1.1, 0.1), 
        labels=[f'{i:.0f}%' for i in np.arange(0, 101, 10)])
    axs[1].set_title('Proportion of missing bat-level observations')
    axs[1].set_ylabel('Missing proportion')
    axs[1].set_xlabel('Time (years)')
    axs[1].set_xticks(
        ticks=np.arange(0, len(df_missing), 12),
        labels=np.arange(df_missing['cal_year'].min(), df_missing['cal_year'].max())
    )
    axs[1].set_xlim(0, len(y))
    axs[1].legend()
    if save_name:
        save_name += '.pdf' if save_name[-4:] != '.pdf' else ''
        plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.show()

    if verbose:
        for f, c, p in zip(features, f_missing, p_missing):
            print(f'{f.ljust(21)}: {c} ({p:.2f}%) missing')

def predictions(
    y_true,
    y_probs,
    cv_splits,
    dates,
    threshold,
    dataset='env',
    save_name=None,
):

    # book keeping
    train_idxs, val_idxs, test_idx = cv_splits
    y_prob_train, y_prob_val, y_prob_test = y_probs
    train_idx = train_idxs[0]
    val_idx = np.concatenate(val_idxs)

    # initialize figure with subplots
    fig, ax = plt.subplots(figsize=(20, 8))

    # axis limits and labels
    ax.set_xlim(-0.5, len(y_true)+0.5)
    ax.set_ylim(-0.01, 1.01)
    ax.set_ylabel('Food shortage probability', fontsize=18)
    ax.set_xlabel('Time (years)', fontsize=18)
            
    # x-axis ticks
    x_ticks = [d[:-2] if d[-2:] == '-6' else '' for d in dates]
    ax.set_xticks(np.arange(len(dates)), x_ticks, rotation=0)
    ax.xaxis.set_label_coords(0.5, -0.1)

    # add vertical year separators
    for i, d in enumerate(dates):
        if d[-2:] == '-0':
            ax.plot([i-0.5, i-0.5], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    # fill ground truth food shortages in green
    for i in range(len(y_true)):
        if y_true[i] == 1:
            ax.fill_between([i-0.5, i+0.5], 0, 1, color='g', alpha=0.5)

    # add vertical threshold line
    ax.hlines(threshold, -0.5, len(y_true)+0.5, 'k', 'dashed', linewidth=2)

    # plot predictions for each split
    ax.plot(train_idx, y_prob_train, '-', c='blue', linewidth=2, markersize=3)
    ax.plot(val_idx, y_prob_val, '-', c='red', linewidth=2, markersize=3)
    ax.plot(test_idx, y_prob_test, '-', c='purple', linewidth=2, markersize=3)

    # label train split
    x_min = train_idx.min()
    x = np.median(train_idxs[0] - x_min)
    if dataset == 'env':
        ax.text(x, -0.09, f'|' + 9*'-' + ' Always train ' + 9*'-' + '|', 
                ha='center', va='bottom', fontsize=14, color='b')
    if dataset == 'bat':
        ax.text(x, -0.09, '|' + 23*'-' + ' Always train ' + 23*'-' + '|', 
                ha='center', va='bottom', fontsize=14, color='b')
    
    # label val splits
    for i, v_idx in enumerate(val_idxs):
        x = np.median(v_idx - x_min)
        if dataset == 'env':
            ax.text(x, -0.09, f'|---- Fold {i+1} ----|', 
                    ha='center', va='bottom', fontsize=14, color='r')
        if dataset == 'bat':
            ax.text(x, -0.09, f'|------------- Fold {i+1} -------------|', 
                    ha='center', va='bottom', fontsize=14, color='r')
    
    # label test split
    x = np.median(test_idx) - 0.5
    if dataset == 'env':
        ax.text(x, -0.09, f'|--------------- Always test --------------|', 
                ha='center', va='bottom', fontsize=14, color='purple')
    if dataset == 'bat':
        ax.text(x, -0.09, f'|----------- Always test -----------|', 
                ha='center', va='bottom', fontsize=14, color='purple')

    # show
    plt.tight_layout(pad=1)
    if save_name is not None:
        save_name += '.pdf' if save_name[-4:] != '.pdf' else ''
        plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.show()


def predictions_subplots(
    y_true,
    y_probs,
    cv_splits,
    dates,
    threshold,
    dataset='env',
    save_name=None,
):
    
    # unpack values
    train_idxs, val_idxs, test_idx = cv_splits
    y_prob_train, y_prob_val, y_prob_test = y_probs
    n_folds = len(val_idxs)

    # one subplot per fold + one for test
    fig, axes = plt.subplots(n_folds+1, 1, figsize=(20, 2 * (n_folds+1)), sharex=True)
    
    if n_folds == 1:
        axes = [axes]

    # x-axis decorations
    x_labels = [d[:-2] if d[-2:] == '-6' else '' for d in dates]
    x_positions = np.arange(len(dates))
    x_min_plot = -0.5
    # x_max_plot = len(y_true) - 0.5
    x_max_plot = test_idx.max() - 0.5

    def draw_year_separators(ax, max_idx):
        for i, d in enumerate(dates):
            if i > max_idx:
                break
            if d[-2:] == '-0':
                ax.plot([i - 0.5, i - 0.5], [0, 1], 'k--', linewidth=1, alpha=0.5)

    #
    # train / val folds
    #

    for i, (tr_idx, v_idx) in enumerate(zip(train_idxs, val_idxs)):

        ax = axes[i]
        fold_max = v_idx.max()

        # fill ground-truth with shading up to fold_max
        labeled = False
        for j in range(tr_idx.min(), fold_max + 1):
            if y_true[j] == 1:
                ax.fill_between([j - 0.5, j + 0.5], 0, 1, 
                                color='g', alpha=0.5, zorder=0, 
                                label='True' if not labeled else None)
                labeled = True

        # plot train predictions
        ax.plot(tr_idx, y_prob_train[i],
                '-', c='blue', linewidth=2, markersize=3, label='Train')

        # plot val predictions
        ax.plot(v_idx, y_prob_val[v_idx - val_idxs[0].min()],
                '-', c='red', linewidth=2, markersize=3, label='Val')

        # horizontal threshold line
        ax.hlines(threshold, -0.5, len(y_true) + 0.5, 
                  color='k', linestyles='dashed', linewidth=2, label='Threshold')

        # draw year separators
        draw_year_separators(ax, fold_max)

        # decorate x-axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=0)

        # decorate y-axis
        ax.set_ylabel(f'Fold {i+1}\nProbability', fontsize=18)
        ax.set_ylim(-0.03, 1.01)
        ax.set_xlim(x_min_plot, x_max_plot)

        # legend
        if dataset == 'env':
            ax.legend(loc='upper left', framealpha=1)
        if dataset == 'bat':
            ax.legend(loc='center', bbox_to_anchor=(-0.075, 0., 0.5, 1), framealpha=1)

    # 
    # train / test fold
    #

    ax = axes[-1]

    tr_idx = train_idxs[-1]
    v_idx = test_idx
    fold_max = test_idx.max()

    labeled = False
    for j in range(tr_idx.min(), fold_max + 1):
        if y_true[j] == 1:
            ax.fill_between([j - 0.5, j + 0.5], 0, 1, 
                            color='g', alpha=0.5, zorder=0, 
                            label='True' if not labeled else None)
            labeled = True

    ax.plot(tr_idx, y_prob_train[-1],
            '-', c='blue', linewidth=2, markersize=3, label='Train')

    ax.plot(v_idx, y_prob_test,
            '-', c='purple', linewidth=2, markersize=3, label='Test')

    ax.hlines(threshold, -0.5, len(y_true) + 0.5, 
                color='k', linestyles='dashed', linewidth=2, label='Threshold')

    draw_year_separators(ax, fold_max)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_xlabel('Time (years)', fontsize=18)
    ax.set_xlim(x_min_plot, x_max_plot)

    ax.set_ylabel(f'Test\nProbability', fontsize=18)
    # ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_ylim(-0.03, 1.01)

    if dataset == 'env':
        ax.legend(loc='upper left', framealpha=1)
    if dataset == 'bat':
        ax.legend(loc='center', bbox_to_anchor=(-0.075, 0., 0.5, 1), framealpha=1)

    #
    # plot cleanup
    #

    plt.tight_layout(pad=1)
    if save_name is not None:
        if not save_name.endswith('.pdf'):
            save_name += '.pdf'
        plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.show()


def shap_beeswarm(
    shap_values, 
    max_display, 
    split, 
    order,
    cmap='coolwarm', 
    save_name=None,
):

    # fix ticks generated from shap library
    def fix_ticks(ax, split='test'):
        y_values, y_labels = ax.get_yticks(), ax.get_yticklabels()
        y_labels = [l.get_text().replace('Sum of', '') for l in y_labels]
        ax.set_yticks(y_values, y_labels)
        ax.set_xlabel(f'SHAP value (impact on model output) for {split} set')

    # visualize shap values
    shap.plots.beeswarm(
        shap_values, max_display=max_display+1, show=False, color=cmap, order=order)
    fix_ticks(plt.gca(), split=split)
    plt.tight_layout()
    if save_name is not None:
        save_name += '.pdf' if save_name[-4:] != '.pdf' else ''
        plt.savefig(save_name, format='pdf')
    plt.show()

def shap_interactions(
    feature_names,
    feature_data,
    feature_shaps,
    feature_interactions,
    individual_splits, 
    interaction_splits,
    rename,
    fontsize=20,
    save_name=None,
):

    # book keeping
    f1, f2 = feature_names
    d1, d2 = feature_data
    s1, s2 = feature_shaps
    interactions = feature_interactions

    # initialize overall figure with 2 x 3 gridspec
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3)

    #
    # plot features and splits for individual features
    #

    # get optimal feature splits for individual features (halves)
    individual_split, count = [], 0
    for f, d, v, s in zip(
        feature_names, feature_data, feature_shaps, individual_splits):

        # plot feature vs shap value with optimal split
        ax = fig.add_subplot(gs[count, 0])
        ax.scatter(d, v, color='k', label='Training values')
        ax.axvline(x=s, color='k', linestyle='--', 
                label=f'Optimal split: {s:.2f}', linewidth=2)
        
        # add decorations
        ax.axhline(y=-0.05, color='k', linestyle='-', linewidth=1, alpha=0.2)
        ax.set_xlabel(rename[f])
        if rename[f] == 'Season':
            ax.set_xticks(
                [0, 1, 2, 3], ['Summer', 'Autumn', 'Winter', 'Spring'])
            ax.set_xlim([-0.2, 3.2])
        ax.set_ylabel('SHAP value', fontsize=fontsize)
        ax.legend(loc='best', fontsize=0.5*fontsize)
        if count == 0: 
            ax.set_title('Optimal feature splits', fontsize=1.2*fontsize)
        
        count += 1

    #
    # plot features and splits for feature interactions
    #

    # scatter the data, colored by strength of interaction
    ax = fig.add_subplot(gs[:, 1:])
    int_sorted = np.sort(interactions)
    int_low = int_sorted[int(0.05*len(int_sorted))]
    int_high = int_sorted[int(0.95*len(int_sorted))]
    ax.scatter(
        d1, d2, c=interactions, cmap='coolwarm', vmin=int_low, vmax=int_high, 
        label='Training values')
    ax.axvline(x=interaction_splits[0], color='k', linestyle='--', linewidth=2, 
            label=f'{rename[f1]} split: {interaction_splits[0]:.2f}')
    ax.axhline(y=interaction_splits[1], color='k', linestyle='-.', linewidth=2, 
            label=f'{rename[f2]} split: {interaction_splits[1]:.2f}')
    ax.legend(loc='best', fontsize=0.75*fontsize)
    ax.get_legend().legend_handles[0].set_color('k')

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(
        vmin=int_low, vmax=int_high))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, aspect=50)
    cbar.ax.set_ylabel('SHAP interaction value', fontsize=fontsize)

    # add extra decorations
    plt.xlabel(rename[f1], fontsize=fontsize)
    if rename[f1] == 'Season':
        plt.xticks([0, 1, 2, 3], ['Summer', 'Autumn', 'Winter', 'Spring'])
        plt.xlim([-0.2, 3.2])
    plt.ylabel(rename[f2], fontsize=fontsize)
    if rename[f2] == 'Season':
        plt.yticks([0, 1, 2, 3], ['Summer', 'Autumn', 'Winter', 'Spring'], 
                   rotation=90, va='center')
        plt.ylim([-0.2, 3.2])
    plt.title('Optimal feature interaction splits', fontsize=1.2*fontsize)

    if save_name is not None:
        save_name += '.pdf' if save_name[-4:] != '.pdf' else ''
        plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.show()

def threshold_model(
    y_true,
    dates,
    feature_names,
    feature_data,
    feature_splits,
    feature_rules,
    train_idx,
    test_idx,
    dataset='env',
    save_name=None,
):

    # book keeping
    f1, f2 = feature_names
    d1, d2 = feature_data
    s1, s2 = feature_splits
    r1, r2 = feature_rules

    # initialize figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 2*8))

    # decorate both axes
    for ax in axs:

        # year labels
        x_ticks = [d[:-2] if d[-2:] == '-6' else '' for d in dates]
        ax.set_xticks(np.arange(len(dates)), x_ticks, rotation=0)
        ax.xaxis.set_label_coords(0.5, -0.1)

        # x-axis limits
        ax.set_xlim(-0.5, len(dates)+0.5)

        # year separators
        for i, d in enumerate(dates):
            if d[-2:] == '-0':
                ax.plot([i-0.5, i-0.5], [-10e10, 10e10], 'k--', linewidth=1, 
                        alpha=0.5)

    #
    # plot threshold model predictions
    #
    
    ax = axs[0]

    # set x/y labels
    ax.set_ylabel('Food shortage probability', fontsize=18, color='black')
    ax.set_ylim(-0.01, 1.01)

    # plot ground truth for train + val
    for i in range(len(y_true)):
        if y_true[i] == 1:
            ax.fill_between([i-0.5, i+0.5], 0, 1, color='g', alpha=0.5)

    # add train/test split labels
    x = np.median(train_idx)
    n = 94 if dataset == 'env' else 95
    ax.text(x, -0.09, '|' + n*'-' + ' Train set ' + n*'-' + '|', ha='center', 
            va='bottom', fontsize=14, color='blue')
    x = np.median(test_idx) - 0.5
    n = 15 if dataset == 'env' else 12
    ax.text(x, -0.09, f'|' + n*'-' + ' Test set ' + n*'-' + '|', ha='center', 
            va='bottom', fontsize=14, color='purple')

    # plot vertical line for train/test split
    ax.vlines(min(test_idx)-0.5, -10e10, 10e10, 'k', 'dashed', linewidth=4)

    # plot classifications for training and test sets
    y_train = r1(d1[train_idx], s1) * r2(d2[train_idx], s2)
    y_test = r1(d1[test_idx], s1) * r2(d2[test_idx], s2)
    ax.plot(train_idx, y_train, '-', c='blue', linewidth=2)
    ax.plot(test_idx, y_test, '-', c='purple', linewidth=2)

    #
    # plot feature data
    #

    # split axes
    ax1 = axs[1]
    ax2 = ax1.twinx()

    # set x/y labels
    ax1.set_xlabel('Time (years)', fontsize=18)

    for i, (ax, f, d, s, r) in enumerate(zip(
        [ax1, ax2], feature_names, feature_data, feature_splits, feature_rules)):

        # plot options
        lw = 1 if dataset == 'env' and f == 'Season' else 2
        c = 'tab:blue' if i == 0 else 'tab:orange'
        alpha = 0.25 if dataset == 'env' and f == 'Season' else 0.5

        # plot feature
        ax.plot(d, '-', c=c, linewidth=lw, zorder=np.inf)
        ax.set_ylabel(f, color=c, fontsize=18)
        ax.set_ylim(
            d.min() - 0.01*(d.max() - d.min()), 
            d.max() + 0.01*(d.max() - d.min()))
        if f == 'Season': 
            ax.set_yticks([0, 1, 2, 3])
        ax.tick_params(axis='y', labelcolor=c)

        # plot horizontal line for feature split
        ax.hlines(s, 0, len(d), c, 'dashed', linewidth=2)

        # add shading
        mask = np.argwhere(r(d, s)).flatten()
        mean = d[mask].mean()
        for i in mask:
            if i == 0 or i == len(d)-1: 
                continue
            y_start = (d[i-1] + d[i]) / 2
            y_end = (d[i] + d[i+1]) / 2
            y1 = [s, s, s] if mean < s else [y_start, d[i], y_end]
            y2 = [y_start, d[i], y_end] if mean < s else [s, s, s]
            ax.fill_between([i-0.5, i, i+0.5], y1, y2, color=c, alpha=alpha)

    plt.tight_layout(pad=2)
    if save_name is not None:
        save_name += '.pdf' if save_name[-4:] != '.pdf' else ''
        plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.show()

