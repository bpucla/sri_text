import argparse
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import shutil
import datetime
import logging
import torch
import sys
import random

def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def set_seed(seed=None):
    if seed is None:
        seed = random.randint(1, 10000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_gpu(gpu, deterministic=True):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


exp_id = get_exp_id(__file__)
output_dir = get_output_dir(exp_id)
copy_source(__file__, output_dir)
logger = setup_logging('main', output_dir)


parser = argparse.ArgumentParser(description='GMM unsupervised clustering')
parser.add_argument('--num', type=int, default=2)
parser.add_argument('--pca_num', type=int, default=4) 
parser.add_argument('--gpu', type=int, default=1) 
parser.add_argument('--seed', type=int, default=888) 


parser.add_argument('--one2one', action="store_true", default=False)

args = parser.parse_args()
logger.info(args)

set_gpu(args.gpu)
set_seed(args.seed)


gmm = GaussianMixture(n_components=args.num, tol=1e-3, max_iter=200, n_init=1, verbose=1)

if args.pca_num > 0:
    pca = PCA(n_components=args.pca_num)

epoch = 8
train_x = torch.load('cache/cache_0_labels/train_data_batch_cache_1', map_location=torch.device('cpu'))
train_true_y = torch.load('cache/cache_0_labels/train_labels_batch_cache_1', map_location=torch.device('cpu'))
train_x = torch.cat(train_x, dim=0).numpy()
train_true_y = [int(l) for labels in train_true_y for l in labels]


test_x = torch.load('cache/cache_0_labels/test_data_batch_cache_1', map_location=torch.device('cpu'))
test_true_y = torch.load('cache/cache_0_labels/test_labels_batch_cache_1', map_location=torch.device('cpu'))
test_x = torch.cat(test_x, dim=0).numpy()
test_true_y = [int(l) for labels in test_true_y for l in labels]


valid_x = torch.load('cache/cache_0_labels/val_data_batch_cache_1', map_location=torch.device('cpu'))
valid_true_y = torch.load('cache/cache_0_labels/val_labels_batch_cache_1', map_location=torch.device('cpu'))
valid_x = torch.cat(valid_x, dim=0).numpy()
valid_true_y = [int(l) for labels in valid_true_y for l in labels]


for _ in range(1):

    logger.info('-------------------------------------------------------------')

    if args.pca_num > 0:
        pca.fit(train_x)

        train_x = pca.transform(train_x)
        valid_x = pca.transform(valid_x)
        test_x = pca.transform(test_x)

    logger.info(train_x.shape)

    logger.info("start fitting gmm on training data")
    gmm.fit(train_x)

    valid_pred_y = gmm.predict(valid_x)

    if args.one2one:
        logger.info("linear assignment")
        cost_matrix = np.zeros((args.num, args.num))

        for i, j in zip(valid_pred_y, valid_true_y):
            cost_matrix[i,j] -= 1

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        # (nsamples, ncomponents)
        valid_score = gmm.predict_proba(valid_x)
        valid_max_index = np.argmax(valid_score, axis=0)
        col_ind = {}
        for i in range(args.num):
            col_ind[i] = valid_true_y[valid_max_index[i]]

    logger.info(col_ind)
    correct = 0.
    for i, j in zip(valid_pred_y, valid_true_y):
        if col_ind[i] == j:
            correct += 1
    logger.info("validation acc {}".format(correct / len(valid_pred_y)))

    test_pred_y = gmm.predict(test_x)
    correct = 0.
    for i, j in zip(test_pred_y, test_true_y):
        if col_ind[i] == j:
            correct += 1
    logger.info("test acc {}".format(correct / len(test_pred_y)))

    train_pred_y = gmm.predict(train_x)
    correct = 0.
    for i, j in zip(train_pred_y, train_true_y):
        if col_ind[i] == j:
            correct += 1
    logger.info("train acc {}".format(correct / len(train_pred_y)))
