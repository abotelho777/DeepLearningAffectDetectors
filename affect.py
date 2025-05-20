import datautility as du
import numpy as np
import tf_network as tfnet
from tf_network import Network, Cost, Normalization, Optimizer
import tensorflow as tf
import os
import time



def apply_clip_sequence(filename, max_rows=None, encoding=None):
    import csv
    import sys

    csvarr = []
    try:
        n_lines = len(open(filename, encoding=encoding).readlines())
    except UnicodeDecodeError:
        encoding = 'utf8'
        n_lines = len(open(filename, encoding=encoding).readlines())
    with open(filename, 'r', errors='replace', encoding=encoding) as f, \
            open(filename[:-4]+'_s.csv', 'w', encoding=encoding) as wr:
        writer = csv.writer(wr, delimiter=',', lineterminator='\n')
        f_lines = csv.reader(f)

        # n_lines = sum(1 for row in f_lines)
        # print(n_lines)
        # n_lines = sum(1 for row in f_lines)
        if max_rows is not None:
            n_lines = max_rows

        output_str = '-- sequencing {}...({}%)'.format(filename, 0)
        sys.stdout.write(output_str)
        sys.stdout.flush()
        old_str = output_str

        header = None
        prevline = None
        ordering = None
        i = 0
        for line in f_lines:
            if len(line) == 0:
                continue
            line = np.array(line)
            na = np.argwhere(np.array(line[:]) == '#N/A').ravel()
            if len(na) > 0:
                line[na] = ''

            na = np.argwhere(np.array(line[:]) == 'NA').ravel()

            if len(na) > 0:
                line[na] = ''

            if i > 0:
                if prevline is None:
                    line[-1] = 0
                    prevline = line
                else:
                    if prevline[5] == line[5] and float(prevline[0]) - float(line[0]) <= 15:
                        line[-1] = prevline[-1]
                    else:
                        line[-1] = str(float(prevline[-1]) + 1)
                    prevline = line
            else:
                line[-1] = 'clip_sequence'
                header = line.ravel()
                ordering = np.argsort(line).ravel()

            row = np.array(line, dtype=str)[ordering]
            row[np.argwhere([k == 'None' for k in row]).ravel()] = ''
            row[np.argwhere([k is None for k in row]).ravel()] = ''
            writer.writerow(row)

            if not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                sys.stdout.write('\r' + (' ' * len(old_str)))
                output_str = '\r-- sequencing {}...({}%)'.format(filename, round((i / n_lines) * 100, 2))
                sys.stdout.write(output_str)
                sys.stdout.flush()
                old_str = output_str

            i += 1
        sys.stdout.write('\r' + (' ' * len(old_str)))
        sys.stdout.write('\r-- sequencing {}...({}%)\n'.format(filename, 100))
        sys.stdout.flush()

    return filename[:-4]+'_s.csv'


def load_affect_model(name='affect'):
    hidden = 200
    layers = 1

    n_cov = 92

    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    print('defining model framework...')
    net = Network(name=name).add_input_layer(n_cov, normalization=Normalization.Z_SCORE)

    for _ in range(layers):
        net.add_lstm_layer(hidden, activation=tf.identity)

    net.begin_multi_output()
    net.add_dropout_layer(4, keep=0.5, activation=tf.nn.softmax)
    net.end_multi_output()

    net.set_default_cost_method(Cost.MULTICLASS_CROSS_ENTROPY)
    net.set_optimizer(Optimizer.ADAM)

    print('building model framework...')
    net.build()

    print('restoring model weights...')
    net.restore_model_weights()
    return net


def train_affect_detectors():
    """
    train_affect_detectors

    uses the aligned affect observations to build, train, and save the affect detector model

    == inputs ==
    - None

    == outputs ==
    - saves model parameters to the '../temp/affect' folder, creating the directory if it does not exist
    """

    outputlabel = 'affect'
    data, headers = du.read_csv('resources/aligned_affect_observations.csv', 100)

    du.print_descriptives(data, headers)

    try:
        seq = dict()
        seq['x'] = np.load('seq_x_' + outputlabel + '.npy', allow_pickle=True)
        seq['y'] = np.load('seq_y_' + outputlabel + '.npy', allow_pickle=True)
        seq['key'] = np.load('seq_k_' + outputlabel + '.npy', allow_pickle=True).reshape((-1,2))
        seq['iden'] = np.load('seq_i_' + outputlabel + '.npy', allow_pickle=True)
    except FileNotFoundError:

        data, headers = du.read_csv('resources/aligned_affect_observations.csv',100)

        du.print_descriptives(data,headers)

        # exit(1)

        key_cols = [5,6]
        cov = list(range(8, 100))
        label_start = 100
        iden = [0,4]
        sortby = 0

        seq = tfnet.format_sequence_from_file('resources/aligned_affect_observations.csv',key_cols,
                                              [list(range(label_start, label_start+4))],cov,iden,sortby)
        # seq = tfnet.format_data(data, key_cols, [list(range(label_start, label_start+4))], cov, sortby, True)

        print('Average Sequence Length: {}'.format(np.mean([len(i) for i in seq['x']])))

        np.save('seq_k_' + outputlabel + '.npy', seq['key'])
        np.save('seq_x_' + outputlabel + '.npy', seq['x'])
        np.save('seq_y_' + outputlabel + '.npy', seq['y'])
        np.save('seq_i_' + outputlabel + '.npy', seq['iden'])

        seq = dict()
        seq['x'] = np.load('seq_x_' + outputlabel + '.npy', allow_pickle=True)
        seq['y'] = np.load('seq_y_' + outputlabel + '.npy', allow_pickle=True)
        seq['key'] = np.load('seq_k_' + outputlabel + '.npy', allow_pickle=True).reshape((-1,2))
        seq['iden'] = np.load('seq_i_' + outputlabel + '.npy', allow_pickle=True)

    max_epochs = 100
    hidden = 200
    batch = 1
    layers = 1
    keep = .5
    step = 1e-4
    threshold = 0
    optimizer = Optimizer.ADAM

    n_cov = len(seq['x'][0][0])
    print('------ {}'.format(n_cov))
    desc = tfnet.describe_multi_label(seq['y'], True)

    tf.set_random_seed(0)
    np.random.seed(0)

    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    net = Network(name='affect').add_input_layer(n_cov, normalization=Normalization.Z_SCORE)

    for _ in range(layers):
        net.add_lstm_layer(hidden, activation=tf.identity)

    net.begin_multi_output()
    for lab_set in desc['n_labels']:
        if lab_set > 1 and lab_set != n_cov:
            activation = tf.nn.softmax
        elif lab_set == 1:
            activation = tf.nn.sigmoid
        else:
            activation = tf.identity

        net.add_dropout_layer(lab_set, keep=keep, activation=activation)
    net.end_multi_output()

    net.set_default_cost_method(Cost.MULTICLASS_CROSS_ENTROPY)
    net.set_optimizer(optimizer)

    net.train(x=seq['x'],
              y=seq['y'],
              step=step,
              use_validation=True,
              max_epochs=max_epochs, threshold=threshold, batch=batch)

    net.save_model_weights()

    p = net.predict(seq['x'])[0]



    du.write_csv(tfnet.flatten_sequence(p, seq['key'], seq['iden']), 'affect_predictions.csv')

    net.session.close()


def apply_affect_detectors(filename, model='affect', net=None):
    """
    apply_affect_detectors

    uses the saved affect detector models (located in ../temp/affect) to produce estimates of
    confusion, engaged concentration, boredom, and frustration.

    == inputs ==
    - data: a 2D numpy array matching the output of the affect feature distiller (WITHOUT column names in the 1st row)
    - headers: a 1D numpy array of the column names corresponding with 'data'

    == outputs ==
    - a 2D numpy array of either 5 or 6 columns corresponding to the problem_log_id, clip (if aggregate is false), and
      an estimate for each of the 4 affective states
    - a 1D list of names corresponding to each of the columns in the first output
    """

    if not os.path.exists('temp/' + model):
        train_affect_detectors()

    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)

    seq_file = apply_clip_sequence(filename)

    data, headers = du.read_csv(seq_file)

    named_cols = np.argwhere([len(h) > 0 for h in headers]).ravel()
    data = data[:, named_cols]
    headers = headers[named_cols]

    du.print_descriptives(data[:, np.argsort(headers).ravel()], headers[np.argsort(headers).ravel()])



    # du.print_descriptives(data,headers)
    # exit(1)

    # col_order = [0,1,2,3,4,5]
    # col_order.extend((np.argsort(headers[6:])+6).tolist())

    data = data[:, np.argsort(headers).ravel()]

    pivot = [92, 96]
    cov = list(range(0, 92))
    iden = [98, 95, 93, 94, 97]
    sortby = 95

    seq = tfnet.format_sequence(data, pivot, None, cov, iden, sortby,verbose=False)

    if net is None:
        net = load_affect_model(model)
        print('model loaded')

    print('generating predictions...')
    p = net.predict(seq['x'])[0]

    print('flattening sequence...')
    pred = np.array(tfnet.flatten_sequence(p, key=seq['key'], identifier=seq['iden']))

    pred_headers = ['row_id', 'clip', 'assignment_id', 'assistment_id', 'problem_id', 'user_id', 'clip_sequence',
                    'p_confusion', 'p_concentration', 'p_boredom', 'p_frustration']
    return pred, pred_headers



if __name__ == '__main__':

    # train_affect_detectors()

    directory = 'resources\\'
    filename = 'arrs_16_19_affect_input_ft.csv'
    #
    start = time.time()

    p, ph = apply_affect_detectors(directory + filename)

    print('writing csv...')
    du.write_csv(p,directory+filename[:-6]+'applied_affect.csv',ph)

    print('Finished in {} seconds.'.format(time.time() - start))
