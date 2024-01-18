import numpy as np
from sklearn import metrics

def get_threshold(y_true, y_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall).clip(1e-8)
    threshold = thresholds[np.argmax(f1)]
    return threshold

def month_to_season(y, dates):

    # month to season mapping
    to_season = {
        12: 1,  1: 1,  2: 1,
        3: 2,  4: 2,  5: 2,
        6: 3,  7: 3,  8: 3,
        9: 4, 10: 4, 11: 4,}

    # book keeping
    months = np.array([d.split('-')[1] for d in dates], dtype=int)
    years = np.array([d.split('-')[0] for d in dates], dtype=int)
    if months.min() == 0: months += 1
    seasons = np.array([to_season[m] for m in months])

    # convert month to season by taking the max over each interval
    y_ssn = []
    for year in np.unique(years):
        for season in np.unique(seasons):
            # if ssn is 0, use jan and feb of this year, dec of last year
            if season == 0:
                mask = (years == year) * ((months == 0) + (months == 1)) + \
                       (years == year-1) * (months == 11)
            else:
                mask = (years == year) * (seasons == season)
            if mask.sum() == 0: 
                continue
            y_ssn.append(y[mask].max().astype(int))
    y_ssn = np.array(y_ssn)
    
    return y_ssn

def get_acc(y_true, y_pred):
    return (y_true == y_pred).mean()

def get_cm(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, normalize='true')

def print_metrics(y_true, y_prob, dates, train_idx, val_idx, test_idx):

    # get threshold
    threshold = get_threshold(y_true[val_idx], y_prob[val_idx])
    y_pred = (y_prob > threshold).astype(int)

    print('Optimal threshold: {:.4f}'.format(threshold))
    print()
    print(30*'-')
    print()

    # book keeping
    y_true_train = y_true[train_idx]
    y_true_val = y_true[val_idx]
    y_true_test = y_true[test_idx]
    y_pred_train = y_pred[train_idx]
    y_pred_val = y_pred[val_idx]
    y_pred_test = y_pred[test_idx]
    dates_train = dates[train_idx]
    dates_val = dates[val_idx]
    dates_test = dates[test_idx]

    # compute month-level metrics
    acc_train = get_acc(y_true_train, y_pred_train)
    acc_val = get_acc(y_true_val, y_pred_val)
    acc_test = get_acc(y_true_test, y_pred_test)
    cm_train = get_cm(y_true_train, y_pred_train)
    cm_val = get_cm(y_true_val, y_pred_val)
    cm_test = get_cm(y_true_test, y_pred_test)

    print('Month-level metrics:')
    print()
    print('Train accuracy:      {:.4f}'.format(acc_train))
    print('Validation accuracy: {:.4f}'.format(acc_val))
    print('Test accuracy:       {:.4f}'.format(acc_test))
    print('Train CM:')
    print(np.round(cm_train, 4))
    print('Validation CM:')
    print(np.round(cm_val, 4))
    print('Test CM:')
    print(np.round(cm_test, 4))
    print()
    print(30*'-')
    print()

    # convert to season-level
    y_true_train_ssn = month_to_season(y_true_train, dates_train)
    y_true_val_ssn = month_to_season(y_true_val, dates_val)
    y_true_test_ssn = month_to_season(y_true_test, dates_test)
    y_pred_train_ssn = month_to_season(y_pred_train, dates_train)
    y_pred_val_ssn = month_to_season(y_pred_val, dates_val)
    y_pred_test_ssn = month_to_season(y_pred_test, dates_test)

    # compute season-level metrics
    acc_train_ssn = get_acc(y_true_train_ssn, y_pred_train_ssn)
    acc_val_ssn = get_acc(y_true_val_ssn, y_pred_val_ssn)
    acc_test_ssn = get_acc(y_true_test_ssn, y_pred_test_ssn)
    cm_train_ssn = get_cm(y_true_train_ssn, y_pred_train_ssn)
    cm_val_ssn = get_cm(y_true_val_ssn, y_pred_val_ssn)
    cm_test_ssn = get_cm(y_true_test_ssn, y_pred_test_ssn)
    
    print('Season-level metrics:')
    print('Train accuracy:      {:.4f}'.format(acc_train_ssn))
    print('Validation accuracy: {:.4f}'.format(acc_val_ssn))
    print('Test accuracy:       {:.4f}'.format(acc_test_ssn))
    print('Train CM:')
    print(np.round(cm_train_ssn, 4))
    print('Validation CM:')
    print(np.round(cm_val_ssn, 4))
    print('Test CM:')
    print(np.round(cm_test_ssn, 4))

