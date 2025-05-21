import numpy as np
import pandas as pd
import joblib
import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Masking, Dense, LSTM, TimeDistributed, Dropout, Normalization
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
import evaluationutility
from preprocess import TARGET_FEATURES, apply_clip_sequence_df, parse_data, pad_data
from sklearn.metrics import roc_curve


def preprocess_data(csv_filename, student_column, clip_column, time_column, clip_sequence_threshold=300):
    data = pd.read_csv(csv_filename)
    data = data.sort_values([student_column, clip_column]).reset_index()
    present_targets = [c for c in TARGET_FEATURES if c in data.columns]

    if present_targets:
        data[TARGET_FEATURES] = data[TARGET_FEATURES].fillna(0)
    data = apply_clip_sequence_df(data, time_col=time_column, group_col=student_column,
                                  threshold=clip_sequence_threshold, sort_columns=False)
    if present_targets:
        data = data[data[TARGET_FEATURES].sum(axis=1) < 2]

    data = data.sort_values([student_column, 'clip_sequence', clip_column]).reset_index()
    parsed_data = parse_data(data, student_column)

    max_expert_length = max(parsed_data['length'])
    padded_expert_input = [pad_data(i, max_expert_length) for i in parsed_data['input']]
    padded_expert_target = None if not present_targets else [pad_data(i, max_expert_length) for i in parsed_data['target']]
    padded_expert_weight = None if not present_targets else [pad_data(i, max_expert_length) for i in parsed_data['weight']]

    return {'input': np.stack(np.array(padded_expert_input)),
            'target': None if not present_targets else np.stack(np.array(padded_expert_target)),
            'weight': None if not present_targets else np.equal(np.stack(np.array(padded_expert_weight)), 1),
            'group': np.array(parsed_data['sid'])}


def train_detectors(X_train, y_train, w_train, save_model=False, model_path='saved_model'):
    keras.backend.clear_session()

    norm = Normalization(axis=-1)
    norm.adapt(X_train)

    lstm = Sequential()
    lstm.add(Masking(mask_value=0.0, input_shape=(X_train[0].shape)))
    lstm.add(norm)
    lstm.add(LSTM(200, activation='tanh', return_sequences=True))
    lstm.add(TimeDistributed(Dropout(0.5)))
    lstm.add(TimeDistributed(Dense(y_train.shape[2], activation='softmax')))
    lstm.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal')

    es = [EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0, restore_best_weights=True)]
    lstm.fit(X_train, y_train, epochs=1000, validation_split=0.3, callbacks=es, verbose=True,
             batch_size=16, shuffle=True)

    yt_pred = lstm.predict(X_train)
    train_a = flatten_padded_sequence(y_train, w_train)
    train_p = flatten_padded_sequence(yt_pred, w_train)

    scaler = MinMaxScaler(feature_range=(0, 1), clip=True).fit(train_p)
    train_p = row_softmax(scaler.transform(train_p))

    thresholds = []
    for idx in range(y_train.shape[2]):
        fpr, tpr, thresh = roc_curve(train_a[:, idx], train_p[:, idx])
        youdens_j = tpr - fpr
        best_idx = np.argmax(youdens_j)
        thresholds.append(thresh[best_idx])

    if save_model:
        os.makedirs(model_path, exist_ok=True)
        lstm.save(os.path.join(model_path, 'lstm_model.h5'))
        joblib.dump(scaler, os.path.join(model_path, 'posthoc_scaler.pkl'))
        joblib.dump(thresholds, os.path.join(model_path, 'thresholds.pkl'))

    return {'model': lstm, 'posthoc_scaler': scaler, 'thresholds': thresholds}


def load_detectors(model_path='saved_model'):
    model_file = os.path.join(model_path, 'lstm_model.h5')
    scaler_file = os.path.join(model_path, 'posthoc_scaler.pkl')
    threshold_file = os.path.join(model_path, 'thresholds.pkl')

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at: {model_file}")
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_file}")
    if not os.path.exists(threshold_file):
        raise FileNotFoundError(f"Threshold file not found at: {scaler_file}")

    model = load_model(model_file)
    scaler = joblib.load(scaler_file)
    thresholds = joblib.load(threshold_file)

    return {'model': model, 'posthoc_scaler': scaler, 'thresholds': thresholds}


def flatten_padded_sequence(sequence, weights=None):
    flat = []
    if weights is None:  # --- keep everything -------------
        for s in sequence:
            flat.append(s)
    else:  # --- drop padded rows ------------
        for s, w in zip(sequence, weights):
            mask = w.reshape(-1)  # ensure 1-D
            flat.append(s[mask, :])
    return np.concatenate(flat, axis=0)


def row_softmax(x: np.ndarray) -> np.ndarray:
    # subtract the per-row max for numerical stability
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def apply_detectors(detector_dict, input_data, weight_data=None):
    lstm = detector_dict['model']
    scaler = detector_dict['posthoc_scaler']

    y_pred = lstm.predict(input_data)

    flat_p = flatten_padded_sequence(y_pred, weight_data)
    flat_p = scaler.transform(flat_p)

    return row_softmax(flat_p)


def evaluate_detectors(input_data, target_data, weight_data, group_data, folds=5):
    gkf = GroupKFold(n_splits=folds)

    auc = []
    conf_auc = []
    conc_auc = []
    bore_auc = []
    frus_auc = []
    kappa = []
    conf_kappa = []
    conc_kappa = []
    bore_kappa = []
    frus_kappa = []

    for train_index, test_index in gkf.split(input_data, target_data, group_data):

        X_train, X_test = input_data[train_index, :, :], input_data[test_index, :, :]
        y_train, y_test = target_data[train_index, :, :], target_data[test_index, :, :]
        w_train, w_test = weight_data[train_index, :], weight_data[test_index, :]

        detector = train_detectors(X_train, y_train, w_train)

        test_p = apply_detectors(detector, input_data=X_test, weight_data=w_test)
        test_a = flatten_padded_sequence(y_test, w_test)

        label_auc = evaluationutility.auc(test_a, test_p, average_over_labels=False)
        label_kappa = evaluationutility.cohen_kappa(test_a, test_p, split=detector['thresholds'], average_over_labels=False)
        auc.append(evaluationutility.auc(test_a, test_p))
        conf_auc.append(label_auc[0])
        conc_auc.append(label_auc[1])
        bore_auc.append(label_auc[2])
        frus_auc.append(label_auc[3])
        kappa.append(evaluationutility.cohen_kappa_multiclass(test_a, test_p))
        conf_kappa.append(label_kappa[0])
        conc_kappa.append(label_kappa[1])
        bore_kappa.append(label_kappa[2])
        frus_kappa.append(label_kappa[3])

    eval = {}
    eval['auc'] = {'all': np.mean(auc),
                   'concentration': np.nanmean(conc_auc),
                   'boredom': np.nanmean(bore_auc),
                   'confusion': np.nanmean(conf_auc),
                   'frustration': np.nanmean(frus_auc)}
    eval['kappa'] = {'all': np.mean(kappa),
                     'concentration': np.nanmean(conc_kappa),
                     'boredom': np.nanmean(bore_kappa),
                     'confusion': np.nanmean(conf_kappa),
                     'frustration': np.nanmean(frus_kappa)}

    return eval



if __name__ == "__main__":

    processed_data = preprocess_data(csv_filename='resources/aligned_affect_observations.csv',
                                     student_column='user_id',
                                     clip_column='clip_id',
                                     time_column='clip_id',
                                     clip_sequence_threshold=15)

    eval = evaluate_detectors(input_data=processed_data['input'],
                              target_data=processed_data['target'],
                              weight_data=processed_data['weight'],
                              group_data=processed_data['group'],   # used for student-level cross validation
                              folds=5)

    print(f'OVERALL - auc: {eval["auc"]["all"]}, kappa: {eval["kappa"]["all"]}')
    print(f'CONCENTRATION - auc: {eval["auc"]["concentration"]}, kappa: {eval["kappa"]["concentration"]}')
    print(f'BOREDOM - auc: {eval["auc"]["boredom"]}, kappa: {eval["kappa"]["boredom"]}')
    print(f'CONFUSION - auc: {eval["auc"]["confusion"]}, kappa: {eval["kappa"]["confusion"]}')
    print(f'FRUSTRATION - auc: {eval["auc"]["frustration"]}, kappa: {eval["kappa"]["frustration"]}')

    # TRAIN AND SAVE MODEL
    train_detectors(processed_data['input'], processed_data['target'], processed_data['weight'], save_model=True)

    # LOAD THE DETECTOR MODEL TRAINED AND SAVED ABOVE
    detector = load_detectors()

    # DETECTOR CONTAINS TWO VALUES:
    # lstm = detector['model']
    # scaler = detector['posthoc_scaler']
    # thresholds = detector['thresholds']  # the learned optimal rounding threshold for each class from the training

    # LOAD AND PREPROCESS THE NEW DATA
    processed_data = preprocess_data(csv_filename='resources/example_unlabeled_data.csv',
                                     student_column='user_id',  # NAME OF THE USER COLUMN
                                     clip_column='clip_id',  # NAME OF THE CLIP/ROW IDENTIFIER COLUMN
                                     time_column='clip_id',  # COLUMN TO BE USED FOR SORTING TEMPORAL SEQUENCES
                                     clip_sequence_threshold=15)  # NUMBER OF UNITS OF TIME (TIME_COLUMN) TO USE AS A
                                                                  # THRESHOLD FOR BREAKING LONG SEQUENCES INTO SHORTER
                                                                  # SEQUENCES (IF SEQUENCES CONTAIN LARGE GAPS IN TIME)
                                                                  # E.G. IF THE VALUE OF TIME_COLUMN FROM ONE ROW TO
                                                                  # ANOTHER WITHIN THE SAME SEQUENCE IS LARGER THAN
                                                                  # CLIP_SEQUENCE_THRESHOLD, A NEW STUDENT SEQUENCE
                                                                  # WILL BE STARTED

    # APPLY THE DETECTORS:
    predictions = apply_detectors(detector, input_data=processed_data['input'])

    print(predictions)






