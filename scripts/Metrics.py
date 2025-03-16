import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_text

def get_threshold(y_true, y_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob)
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
):
    # book keeping
    train = train_idx is not None
    val = val_idx is not None
    test = test_idx is not None

    # get threshold
    if threshold is None and val:
        threshold = get_threshold(y_true[val_idx], y_prob[val_idx])
        y_pred = (y_prob > threshold).astype(int)
    elif threshold is None and not val and train:
        threshold = get_threshold(y_true[train_idx], y_prob[train_idx])
        y_pred = (y_prob > threshold).astype(int)
    else:
        # user-specified threshold
        y_pred = (y_prob > threshold).astype(int)

    print('Optimal threshold: {:.4f}'.format(threshold))
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
        acc_train = get_acc(y_true_train, y_pred_train)
        cm_train = get_cm(y_true_train, y_pred_train)
        print('Train accuracy:      {:.4f}'.format(acc_train))
        print('Train CM:')
        print(np.round(cm_train, 4))
        print()

    if val:
        y_true_val = y_true[val_idx]
        y_pred_val = y_pred[val_idx]
        dates_val = dates[val_idx]
        acc_val = get_acc(y_true_val, y_pred_val)
        cm_val = get_cm(y_true_val, y_pred_val)
        print('Validation accuracy: {:.4f}'.format(acc_val))
        print('Validation CM:')
        print(np.round(cm_val, 4))
        print()

    if test:
        y_true_test = y_true[test_idx]
        y_pred_test = y_pred[test_idx]
        dates_test = dates[test_idx]
        acc_test = get_acc(y_true_test, y_pred_test)
        cm_test = get_cm(y_true_test, y_pred_test)
        print('Test accuracy:       {:.4f}'.format(acc_test))
        print('Test CM:')
        print(np.round(cm_test, 4))
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
        acc_train_ssn = get_acc(y_true_train_ssn, y_pred_train_ssn)
        cm_train_ssn = get_cm(y_true_train_ssn, y_pred_train_ssn)
        print('Train accuracy:      {:.4f}'.format(acc_train_ssn))
        print('Train CM:')
        print(np.round(cm_train_ssn, 4))
        print()

    if val:
        y_true_val_ssn = month_to_season(y_true_val, dates_val)
        y_pred_val_ssn = month_to_season(y_pred_val, dates_val)
        acc_val_ssn = get_acc(y_true_val_ssn, y_pred_val_ssn)
        cm_val_ssn = get_cm(y_true_val_ssn, y_pred_val_ssn)
        print('Validation accuracy: {:.4f}'.format(acc_val_ssn))
        print('Validation CM:')
        print(np.round(cm_val_ssn, 4))
        print()

    if test:
        y_true_test_ssn = month_to_season(y_true_test, dates_test)
        y_pred_test_ssn = month_to_season(y_pred_test, dates_test)
        acc_test_ssn = get_acc(y_true_test_ssn, y_pred_test_ssn)
        cm_test_ssn = get_cm(y_true_test_ssn, y_pred_test_ssn)
        print('Test accuracy:       {:.4f}'.format(acc_test_ssn))
        print('Test CM:')
        print(np.round(cm_test_ssn, 4))
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
        print('Train event-based precision:  {:.4f}'.format(train_event_stats['event_precision']))
        print('Train event-based recall:     {:.4f}'.format(train_event_stats['event_recall']))
        print('Train mean start error:       {:.4f}'.format(train_event_stats['mean_start_error']))
        print('Train mean end error:         {:.4f}'.format(train_event_stats['mean_end_error']))
        print('Train mean duration error:    {:.4f}'.format(train_event_stats['mean_duration_error']))
        print()

    if val:
        val_event_stats = get_event_metrics(y_true_val, y_pred_val, dates_val)
        print('Validation event-based precision:  {:.4f}'.format(val_event_stats['event_precision']))
        print('Validation event-based recall:     {:.4f}'.format(val_event_stats['event_recall']))
        print('Validation mean start error:       {:.4f}'.format(val_event_stats['mean_start_error']))
        print('Validation mean end error:         {:.4f}'.format(val_event_stats['mean_end_error']))
        print('Validation mean duration error:    {:.4f}'.format(val_event_stats['mean_duration_error']))
        print()

    if test:
        test_event_stats = get_event_metrics(y_true_test, y_pred_test, dates_test)
        print('Test event-based precision:        {:.4f}'.format(test_event_stats['event_precision']))
        print('Test event-based recall:           {:.4f}'.format(test_event_stats['event_recall']))
        print('Test mean start error:             {:.4f}'.format(test_event_stats['mean_start_error']))
        print('Test mean end error:               {:.4f}'.format(test_event_stats['mean_end_error']))
        print('Test mean duration error:          {:.4f}'.format(test_event_stats['mean_duration_error']))
        print()


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