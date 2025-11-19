import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_text

def precision_recall(y_true, y_prob, weight=None):
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true, y_prob, sample_weight=weight)
    return precision, recall, thresholds

def get_threshold(y_true, y_prob, weight=None):
    if y_true.sum() == 0:
        return np.nan
    precision, recall, thresholds = precision_recall(
        y_true, y_prob, weight=weight)
    f1 = 2 * (precision * recall) / (precision + recall).clip(1e-8)
    threshold = thresholds[np.argmax(f1)]
    return threshold

def month_to_season(y, dates):

    # month to season mapping
    to_season = {
        12: 1,  1: 1,  2: 1,
        3: 2,   4: 2,  5: 2,
        6: 3,   7: 3,  8: 3,
        9: 4,  10: 4, 11: 4,
    }

    # book keeping
    months = np.array([int(d.split('-')[1]) for d in dates], dtype=int)
    years = np.array([int(d.split('-')[0]) for d in dates], dtype=int)

    # ensure no zero-based months
    if months.min() == 0:
        months += 1

    seasons = np.array([to_season[m] for m in months])

    # convert month to season by taking the max over each interval
    y_ssn = []
    for year in np.unique(years):
        for season in np.unique(seasons):
            mask = (years == year) & (seasons == season)
            if mask.sum() == 0:
                continue
            # if there's at least one "1" in the block, label the season as 1
            y_ssn.append(y[mask].max().astype(int))

    return np.array(y_ssn)

def get_acc(y_true, y_pred):
    return (y_true == y_pred).mean()

def get_cm(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, normalize='true')

def get_blocks(y, dates):
    
    # initialize
    blocks = []
    in_block = False
    start_idx = None

    # iterate over y
    for i, val in enumerate(y):

        # start of a block
        if val == 1 and not in_block:
            in_block = True
            start_idx = i

        # end of a block
        elif val == 0 and in_block:
            
            end_idx = i - 1
            blocks.append({
                'start_index': start_idx,
                'end_index': end_idx,
                'start_date': dates[start_idx],
                'end_date': dates[end_idx],
                'duration': (end_idx - start_idx + 1)
            })
            in_block = False

    # if the last value was part of a block, close it
    if in_block:
        end_idx = len(y) - 1
        blocks.append({
            'start_index': start_idx,
            'end_index': end_idx,
            'start_date': dates[start_idx],
            'end_date': dates[end_idx],
            'duration': (end_idx - start_idx + 1)
        })

    return blocks

def get_event_metrics(y_true, y_pred, dates):
    
    # get blocks
    true_blocks = get_blocks(y_true, dates)
    pred_blocks = get_blocks(y_pred, dates)

    print(f'True blocks: {len(true_blocks)}, Predicted blocks: {len(pred_blocks)}')
    
    # handle trivial cases
    if len(true_blocks) == 0 and len(pred_blocks) == 0:
        return {
            'event_precision': 1.0,  
            'event_recall': 1.0,
            'mean_start_error': 0.0,
            'mean_end_error': 0.0,
            'mean_duration_error': 0.0
        }
    elif len(pred_blocks) == 0:
        return {
            'event_precision': 0.0,
            'event_recall': 0.0,
            'mean_start_error': np.nan,
            'mean_end_error': np.nan,
            'mean_duration_error': np.nan
        }
    elif len(true_blocks) == 0:
        return {
            'event_precision': 0.0,
            'event_recall': 0.0,
            'mean_start_error': np.nan,
            'mean_end_error': np.nan,
            'mean_duration_error': np.nan
        }

    # find overlapping blocks
    matched_pairs = []
    for tb in true_blocks:
        t_range = set(range(tb['start_index'], tb['end_index'] + 1))
        overlapping_preds = []
        for pb in pred_blocks:
            p_range = set(range(pb['start_index'], pb['end_index'] + 1))
            if t_range & p_range: # non-empty intersection means overlap
                overlapping_preds.append(pb)
        if overlapping_preds:
            # consider the first overlapping predicted block as the matched one.
            matched_pairs.append((tb, overlapping_preds[0]))

    # compute event-level metrics
    matched_true = set()
    matched_preds = set()
    for (tb, pb) in matched_pairs:
        matched_true.add((tb['start_index'], tb['end_index']))
        matched_preds.add((pb['start_index'], pb['end_index']))
    all_true_ids = set((b['start_index'], b['end_index']) for b in true_blocks)
    all_pred_ids = set((b['start_index'], b['end_index']) for b in pred_blocks)

    # fraction of predicted blocks that match at least one true block
    event_precision = len(matched_preds) / len(all_pred_ids) 

    # fraction of true blocks that match at least one predicted block
    event_recall = len(matched_true) / len(all_true_ids)

    # initialize
    start_errors = []
    end_errors = []
    duration_errors = []

    # compute differences in start, end, and duration for matched pairs only
    for (tb, pb) in matched_pairs:
        start_errors.append(abs(tb['start_index'] - pb['start_index']))
        end_errors.append(abs(tb['end_index'] - pb['end_index']))
        duration_errors.append(abs(tb['duration'] - pb['duration']))

    # compute means
    if len(matched_pairs) == 0:
        mean_start_error = np.nan
        mean_end_error = np.nan
        mean_duration_error = np.nan
    else:
        mean_start_error = np.mean(start_errors)
        mean_end_error = np.mean(end_errors)
        mean_duration_error = np.mean(duration_errors)

    result = {
        'event_precision': event_precision,
        'event_recall': event_recall,
        'mean_start_error': mean_start_error,
        'mean_end_error': mean_end_error,
        'mean_duration_error': mean_duration_error
    }

    return result

def print_metrics(
    y_true,
    y_prob,
    dates,
    threshold=None,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    decimal=4,
):
    # book keeping
    train = train_idx is not None
    val = val_idx is not None
    test = test_idx is not None

    # get threshold
    if threshold is None and val:
        threshold = get_threshold(y_true[val_idx], y_prob[val_idx])
        y_pred = (y_prob >= threshold).astype(int)
    elif threshold is None and not val and train:
        threshold = get_threshold(y_true[train_idx], y_prob[train_idx])
        y_pred = (y_prob >= threshold).astype(int)
    else:
        # user-specified threshold
        y_pred = (y_prob >= threshold).astype(int)

    print('Optimal threshold: {}'.format(threshold))
    print('Rounded threshold: {}'.format(np.round(threshold, decimal)))
    print()
    print('-' * 30)
    print()

    # 
    # month-level
    # 

    print('Month-level metrics:')
    print()

    if train:
        y_true_train = y_true[train_idx]
        y_pred_train = y_pred[train_idx]
        dates_train = dates[train_idx]
        acc_train = np.round(get_acc(y_true_train, y_pred_train), decimal)
        cm_train = np.round(get_cm(y_true_train, y_pred_train), decimal)
        print('Train accuracy:      {}'.format(acc_train))
        print('Train CM:')
        print(cm_train)
        print()

    if val:
        y_true_val = y_true[val_idx]
        y_pred_val = y_pred[val_idx]
        dates_val = dates[val_idx]
        acc_val = np.round(get_acc(y_true_val, y_pred_val), decimal)
        cm_val = np.round(get_cm(y_true_val, y_pred_val), decimal)
        print('Validation accuracy: {}'.format(acc_val))
        print('Validation CM:')
        print(cm_val)
        print()

    if test:
        y_true_test = y_true[test_idx]
        y_pred_test = y_pred[test_idx]
        dates_test = dates[test_idx]
        acc_test = np.round(get_acc(y_true_test, y_pred_test), decimal)
        cm_test = np.round(get_cm(y_true_test, y_pred_test), decimal)
        print('Test accuracy:       {}'.format(acc_test))
        print('Test CM:')
        print(cm_test)
        print()

    print('-' * 30)
    print()

    # 
    # season-level
    # 

    print('Season-level metrics:')
    print()

    if train:
        y_true_train_ssn = month_to_season(y_true_train, dates_train)
        y_pred_train_ssn = month_to_season(y_pred_train, dates_train)
        acc_train_ssn = np.round(get_acc(y_true_train_ssn, y_pred_train_ssn), decimal)
        cm_train_ssn = np.round(get_cm(y_true_train_ssn, y_pred_train_ssn), decimal)
        print('Train accuracy:      {}'.format(acc_train_ssn))
        print('Train CM:')
        print(cm_train_ssn)
        print()

    if val:
        y_true_val_ssn = month_to_season(y_true_val, dates_val)
        y_pred_val_ssn = month_to_season(y_pred_val, dates_val)
        acc_val_ssn = np.round(get_acc(y_true_val_ssn, y_pred_val_ssn), decimal)
        cm_val_ssn = np.round(get_cm(y_true_val_ssn, y_pred_val_ssn), decimal)
        print('Validation accuracy: {}'.format(acc_val_ssn))
        print('Validation CM:')
        print(cm_val_ssn)
        print()

    if test:
        y_true_test_ssn = month_to_season(y_true_test, dates_test)
        y_pred_test_ssn = month_to_season(y_pred_test, dates_test)
        acc_test_ssn = np.round(get_acc(y_true_test_ssn, y_pred_test_ssn), decimal)
        cm_test_ssn = np.round(get_cm(y_true_test_ssn, y_pred_test_ssn), decimal)
        print('Test accuracy:       {}'.format(acc_test_ssn))
        print('Test CM:')
        print(cm_test_ssn)
        print()

    print('-' * 30)
    print()

    # 
    # event-level
    # 
    
    print('Event-level (contiguous block) metrics:')
    print()

    if train:
        train_event_stats = get_event_metrics(y_true_train, y_pred_train, dates_train)
        print('Train event-based precision:  {}'.format(
            np.round(train_event_stats['event_precision'], decimal)))
        print('Train event-based recall:     {}'.format(
            np.round(train_event_stats['event_recall'], decimal)))
        print('Train mean start error:       {}'.format(
            np.round(train_event_stats['mean_start_error'], decimal)))
        print('Train mean end error:         {}'.format(
            np.round(train_event_stats['mean_end_error'], decimal)))
        print('Train mean duration error:    {}'.format(
            np.round(train_event_stats['mean_duration_error'], decimal)))
        print()

    if val:
        val_event_stats = get_event_metrics(y_true_val, y_pred_val, dates_val)
        print('Validation event-based precision:  {}'.format(
            np.round(val_event_stats['event_precision'], decimal)))
        print('Validation event-based recall:     {}'.format(
            np.round(val_event_stats['event_recall'], decimal)))
        print('Validation mean start error:       {}'.format(
            np.round(val_event_stats['mean_start_error'], decimal)))
        print('Validation mean end error:         {}'.format(
            np.round(val_event_stats['mean_end_error'], decimal)))
        print('Validation mean duration error:    {}'.format(
            np.round(val_event_stats['mean_duration_error'], decimal)))
        print()

    if test:
        test_event_stats = get_event_metrics(y_true_test, y_pred_test, dates_test)
        print('Test event-based precision:        {}'.format(
            np.round(test_event_stats['event_precision'], decimal)))
        print('Test event-based recall:           {}'.format(
            np.round(test_event_stats['event_recall'], decimal)))
        print('Test mean start error:             {}'.format(
            np.round(test_event_stats['mean_start_error'], decimal)))
        print('Test mean end error:               {}'.format(
            np.round(test_event_stats['mean_end_error'], decimal)))
        print('Test mean duration error:          {}'.format(
            np.round(test_event_stats['mean_duration_error'], decimal)))
        print()
    

def _split_idx(train_idx, val_idx, test_idx, split):
    return {"train": train_idx, "val": val_idx, "test": test_idx}[split]

def _summarize_block(y_true, y_pred, dates, decimal=1, pct=True):
    scale = 100.0 if pct else 1.0
    # month level
    acc_m = get_acc(y_true, y_pred) * scale
    cm_m  = get_cm(y_true, y_pred)      # rows sum to 1
    tpr_m = cm_m[1, 1] * scale          # sensitivity
    tnr_m = cm_m[0, 0] * scale          # specificity
    # season level
    y_true_s = month_to_season(y_true, dates)
    y_pred_s = month_to_season(y_pred, dates)
    acc_s = get_acc(y_true_s, y_pred_s) * scale
    cm_s  = get_cm(y_true_s, y_pred_s)
    tpr_s = cm_s[1, 1] * scale
    tnr_s = cm_s[0, 0] * scale
    # event level
    ev = get_event_metrics(y_true, y_pred, dates)
    P  = ev["event_precision"] * scale if ev["event_precision"] is not None else np.nan
    R  = ev["event_recall"]    * scale if ev["event_recall"]    is not None else np.nan
    SE = ev["mean_start_error"]
    EE = ev["mean_end_error"]
    DE = ev["mean_duration_error"]
    vals = dict(
        MA=acc_m, MTPR=tpr_m, MTNR=tnr_m,
        SA=acc_s, STPR=tpr_s, STNR=tnr_s,
        P=P, R=R, SE=SE, EE=EE, DE=DE
    )
    # round
    for k, v in vals.items():
        if v is None: continue
        if np.isnan(v): continue
        vals[k] = np.round(v, decimal)
    return vals

def generate_table(y_true, y_prob, dates,
                  split="val",
                  train_idx=None, val_idx=None, test_idx=None,
                  threshold=None, decimal=1, pct=True,
                  sep="\t", return_string=True):
    """Outputs metrics in this column order:
       MA, MTPR, MTNR, SA, STPR, STNR, P, R, SE, EE, DE
    """
    # choose split indices
    idx = _split_idx(train_idx, val_idx, test_idx, split)
    assert idx is not None, f"{split} indices are required"
    # choose threshold
    if threshold is None:
        ref_idx = val_idx if val_idx is not None else train_idx
        assert ref_idx is not None, "need train_idx or val_idx to tune threshold"
        threshold = get_threshold(y_true[ref_idx], y_prob[ref_idx])
    y_pred = (y_prob >= threshold).astype(int)

    # slice this split
    yt, yp, dt = y_true[idx], y_pred[idx], dates[idx]
    vals = _summarize_block(yt, yp, dt, decimal=decimal, pct=pct)

    if return_string:
        order = ["MA","MTPR","MTNR","SA","STPR","STNR","P","R","SE","EE","DE"]
        s = sep.join("" if (k in vals and (vals[k] is None or np.isnan(vals[k])))
                     else str(vals[k]) for k in order)
        print(s)  # one clean line for copy/paste
        return vals, s
    return vals


def optimal_shap_splits(
    x,
    y,
    feature_names,
    weights=None,
    verbose=False,
):

    # train decision tree
    n_features = x.shape[1]
    dt = DecisionTreeClassifier(max_depth=n_features)
    if weights is not None:
        dt.fit(x, y, sample_weight=weights)
    else:
        dt.fit(x, y)
    dt_txt = export_text(dt, feature_names=feature_names)
    if verbose: 
        print(dt_txt)

    # parse decision tree
    dt_txt = dt_txt.split('\n')
    dt_txt = [t for t in dt_txt if ' <= ' in t]
    names, splits = [], []
    for i in range(len(dt_txt)):
        name, value = dt_txt[i].split(' <= ')
        name = name.split('|--- ')[-1]
        value = float(value.strip())
        names.append(name)
        splits.append(value)

    # aggregate splits
    names, splits = np.array(names), np.array(splits)
    splits = np.array([splits[names == n].max() for n in feature_names])

    return names, splits