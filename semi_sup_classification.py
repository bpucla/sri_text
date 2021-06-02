#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil
import copy
import logging
import datetime
import itertools

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np
import h5py
import time
from optim_n2n import OptimN2N
from data import Dataset
import utils

from preprocess_text import Indexer

import pygrid

def parse_args():

    parser = argparse.ArgumentParser()

    # Input data
    parser.add_argument('--dataset', default='yelp')
    parser.add_argument('--train_data', type=str, default='datasets/short_yelp_data/short_yelp.train.txt')
    parser.add_argument('--val_data', type=str, default='datasets/short_yelp_data/short_yelp.valid.txt')
    parser.add_argument('--test_data', type=str, default='datasets/short_yelp_data/short_yelp.test.txt')
    parser.add_argument('--vocab_file', type=str, default='datasets/short_yelp_data/vocab.txt')
    parser.add_argument('--label', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_from', default='')
    parser.add_argument('--use_cache', type=bool, default=True)


    # SRI options
    parser.add_argument('--z_n_iters', type=int, default=20)
    parser.add_argument('--z_step_size', type=float, default=0.8)
    parser.add_argument('--z_with_noise', type=int, default=0)
    parser.add_argument('--prior_sigma', type=float, default=1.0)
    parser.add_argument('--num_z_samples', type=int, default=10)
    parser.add_argument('--noise_temp', type=float, default=1.0)

    # Model options
    parser.add_argument('--latent_dim', default=32, type=int)
    parser.add_argument('--enc_word_dim', default=256, type=int)
    parser.add_argument('--enc_h_dim', default=256, type=int)
    parser.add_argument('--enc_num_layers', default=1, type=int)
    parser.add_argument('--dec_word_dim', default=128, type=int)
    parser.add_argument('--dec_h_dim', default=512, type=int)
    parser.add_argument('--dec_num_layers', default=1, type=int)
    parser.add_argument('--dec_dropout', default=0.5, type=float)
    parser.add_argument('--model', default='abp', type=str, choices=['abp', 'vae'])
    parser.add_argument('--train_n2n', default=1, type=int)
    parser.add_argument('--train_kl', default=1, type=int)
    parser.add_argument('--train_froim', default='', type=str)

    # Optimization options
    parser.add_argument('--checkpoint_dir', default='models/ptb')
    parser.add_argument('--slurm', default=0, type=int)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('--num_epochs', default=12, type=int)
    parser.add_argument('--min_epochs', default=15, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--svi_steps', default=10, type=int)
    parser.add_argument('--svi_lr1', default=1, type=float)
    parser.add_argument('--svi_lr2', default=1, type=float)
    parser.add_argument('--eps', default=1e-5, type=float)
    parser.add_argument('--decay', default=0, type=int)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_grad_norm', default=5, type=float)
    parser.add_argument('--svi_max_grad_norm', default=5, type=float)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--seed', default=8485, type=int)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--kl_every', type=int, default=20)
    parser.add_argument('--test', type=int, default=0)

    # Classification options
    parser.add_argument('--cls_train_data', type=str, default='datasets/short_yelp_data/short_yelp.train.100.txt')
    parser.add_argument("--discriminator", type=str, default="linear")
    parser.add_argument('--ncluster', type=int, default=2)
    parser.add_argument('--cls_epochs', type=int, default=100)
    parser.add_argument('--cls_update_every', type=int, default=1)
    parser.add_argument('--cls_test_nepoch', type=int, default=5)
    parser.add_argument("--cls_load_best_epoch", type=int, default=0)
    parser.add_argument('--cls_momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--cls_lr', default=1., type=float)
    parser.add_argument('--M', default=5, type=int)

    return parser.parse_args()


def create_args_grid():
    # TODO add your enumeration of parameters here

    # step sizes = [0.2, 0.3, 0.4, 0.5], number of steps = [15, 20], z_with_noise = [0, 1], warmup=[10, 15]

    z_step_size = [0.4]
    z_n_iters = [40]
    z_with_noise = [0]
    warmup = [6]
    M = [200]
    noise_temp = [0.0001]

    args_list = [z_step_size, z_n_iters, z_with_noise, warmup, M, noise_temp]

    opt_list = []
    for i, args in enumerate(itertools.product(*args_list)):
        opt_job = {'job_id': int(i), 'status': 'open'}
        opt_args = {
            'z_step_size': args[0],
            'z_n_iters': args[1],
            'z_with_noise': args[2],
            'warmup': args[3],
            'M': args[4],
        }
        # TODO add your result metric here
        opt_result = {'val_nll':0., 'val_rec_abp':0., 'val_kl_abp':0., 'val_ppl_bound_abp':0., 'test_nll':0., 'test_rec_abp':0., 'test_kl_abp':0., 'test_ppl_bound_abp':0., 'cls_loss':0., 'cls_acc':0., 'cls_loss_2':0., 'cls_acc_2':0.}

        # opt_list += [{**opt_job, **opt_args, **opt_result}]
        opt_list += [merge_dicts(opt_job, opt_args, opt_result)]

    return opt_list


def update_job_result(job_opt, job_stats):
    # TODO add your result metric here
    job_opt['val_nll'] = job_stats['val_nll']
    job_opt['val_rec_abp'] = job_stats['val_rec_abp']
    job_opt['val_kl_abp'] = job_stats['val_kl_abp']
    job_opt['val_ppl_bound_abp'] = job_stats['val_ppl_bound_abp']
    job_opt['test_nll'] = job_stats['test_nll']
    job_opt['test_rec_abp'] = job_stats['test_rec_abp']
    job_opt['test_kl_abp'] = job_stats['test_kl_abp']
    job_opt['test_ppl_bound_abp'] = job_stats['test_ppl_bound_abp']
    job_opt['cls_loss'] = job_stats['cls_loss']
    job_opt['cls_acc'] = job_stats['cls_acc']
    job_opt['cls_loss_2'] = job_stats['cls_loss_2']
    job_opt['cls_acc_2'] = job_stats['cls_acc_2']

##------------------------------------------------------------------------------------------------------------------##


def _dropout_mask1(x, p=0.5):
    mask = torch.ones_like(x)
    index2drop = torch.rand_like(mask) < p
    mask[index2drop] = 0.
    return mask * (1. / (1. - p))


def _dropout_mask2(sent, last_dim, p=0.5):
    batch_size, seq_len = sent.size()
    seq_len = seq_len - 1
    mask = torch.ones((batch_size, seq_len, last_dim), device=sent.device, dtype=torch.float)
    index2drop = torch.rand_like(mask) < p
    mask[index2drop] = 0.
    return mask * (1. / (1. - p))


def dropout_mask(sent, embedding_size, hidden_dim, p=0.5):
    in_mask = _dropout_mask2(sent, embedding_size, p=p)
    out_mask = _dropout_mask2(sent, hidden_dim, p=p)
    return in_mask, out_mask


def dropout(x, p=0.5, mask=None, training=True):
    if mask is None:
        mask = _dropout_mask1(x, p)
    if training:
        return x * mask
    else:
        return x


##------------------------------------------------------------------------------------------------------------------##

class RNNVAE(nn.Module):
    def __init__(self, args, vocab_size=10000,
                 enc_word_dim=512,
                 enc_h_dim=1024,
                 enc_num_layers=1,
                 dec_word_dim=512,
                 dec_h_dim=1024,
                 dec_num_layers=1,
                 dec_dropout=0.5,
                 latent_dim=32,
                 max_sequence_length=40,
                 mode='savae'):
        super(RNNVAE, self).__init__()
        self.args = args
        self.enc_h_dim = enc_h_dim
        self.enc_num_layers = enc_num_layers
        self.dec_h_dim = dec_h_dim
        self.dec_num_layers = dec_num_layers
        self.embedding_size = dec_word_dim
        self.latent_dim = latent_dim
        self.max_sequence_length = max_sequence_length

        self.enc_word_vecs = nn.Embedding(vocab_size, enc_word_dim)
        self.latent_linear_mean = nn.Linear(enc_h_dim, latent_dim)
        self.latent_linear_logvar = nn.Linear(enc_h_dim, latent_dim)
        self.enc_rnn = nn.LSTM(enc_word_dim, enc_h_dim, num_layers=enc_num_layers,
                                batch_first=True)
        self.enc = nn.ModuleList([self.enc_word_vecs, self.enc_rnn,
                                    self.latent_linear_mean, self.latent_linear_logvar])

        self.dec_word_vecs = nn.Embedding(vocab_size, dec_word_dim)
        dec_input_size = dec_word_dim
        dec_input_size += latent_dim
        self.dec_rnn = nn.LSTM(dec_input_size, dec_h_dim, num_layers=dec_num_layers,
                               batch_first=True)
        # self.input_dropout = nn.Dropout(dec_dropout)
        # self.out_dropout = nn.Dropout(dec_dropout)
        self.dec_linear = nn.Sequential(*[nn.Linear(dec_h_dim, vocab_size),
                                          nn.LogSoftmax(dim=-1)])
        self.dec = nn.ModuleList([self.dec_word_vecs, self.dec_rnn, self.dec_linear])
        if latent_dim > 0:
            self.latent_hidden_linear = nn.Linear(latent_dim, dec_h_dim)
            self.dec.append(self.latent_hidden_linear)

    def _enc_forward(self, sent):
        word_vecs = self.enc_word_vecs(sent)
        h0 = Variable(torch.zeros(self.enc_num_layers, word_vecs.size(0),
                                  self.enc_h_dim).type_as(word_vecs.data))
        c0 = Variable(torch.zeros(self.enc_num_layers, word_vecs.size(0),
                                  self.enc_h_dim).type_as(word_vecs.data))
        enc_h_states, _ = self.enc_rnn(word_vecs, (h0, c0))
        enc_h_states_last = enc_h_states[:, -1]
        mean = self.latent_linear_mean(enc_h_states_last)
        logvar = self.latent_linear_logvar(enc_h_states_last)
        return mean, logvar

    def _reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = Variable(torch.cuda.FloatTensor(std.size()).normal_(0, 1))
        return z.mul(std) + mean

    def infer_z(self, z, sent, beta=1., training=True, in_mask=None, out_mask=None):
        args = self.args
        target = sent.detach().clone()
        target = target[:, 1:]
        z_grads_norm = []
        # in_mask, out_mask = dropout_mask(sent, self.embedding_size, self.dec_h_dim, p=0.5)
        for i in range(args.z_n_iters):
            # z = torch.autograd.Variable(z.detach().clone(), requires_grad=True)
            z = z.detach().clone()
            z.requires_grad = True
            assert z.grad is None
            logp = self._dec_forward(sent, z, training=training, in_mask=in_mask, out_mask=out_mask)  # TODO: turn off dropout in inference?
            logp = logp.view(-1, logp.size(2))
            nll = F.nll_loss(logp, target.reshape(-1), reduction='sum', ignore_index=0)  # TODO remove hard-coding
            nll.backward()
            z_grad = z.grad.detach().clone()
            z = z - 0.5 * args.z_step_size * args.z_step_size * (beta * z + z.grad)
            if args.z_with_noise:
                z += args.noise_temp * args.z_step_size * torch.randn_like(z)

            z_grads_norm.append(torch.norm(z_grad, dim=0).mean().cpu().numpy())

        z = z.detach()

        return z, z_grads_norm

    def _dec_forward(self, sent, q_z, init_h=True, training=True, in_mask=None, out_mask=None):
        self.word_vecs = F.dropout(self.dec_word_vecs(sent[:, :-1]), p=0.5, training=training)
        if init_h:
            self.h0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim).type_as(self.word_vecs.data), requires_grad=False)
            self.c0 = Variable(torch.zeros(self.dec_num_layers, self.word_vecs.size(0), self.dec_h_dim).type_as(self.word_vecs.data), requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()

        if q_z is not None:
            q_z_expand = q_z.unsqueeze(1).expand(self.word_vecs.size(0),
                                                 self.word_vecs.size(1), q_z.size(1))
            dec_input = torch.cat([self.word_vecs, q_z_expand], 2)
        else:
            dec_input = self.word_vecs
        if q_z is not None:
            self.h0[-1] = self.latent_hidden_linear(q_z)
        memory, _ = self.dec_rnn(dec_input, (self.h0, self.c0))
        dec_linear_input = memory.contiguous()
        dec_linear_input = F.dropout(dec_linear_input, p=0.5, training=training)
        preds = self.dec_linear(dec_linear_input.view(
            self.word_vecs.size(0) * self.word_vecs.size(1), -1)).view(
            self.word_vecs.size(0), self.word_vecs.size(1), -1)
        return preds

    def inference(self, device, sos_idx, n=4, z=None, init_h=True, training=True):
        if z is None:
            batch_size = n
            z = torch.randn([batch_size, self.latent_dim], device=device)
        else:
            batch_size = z.size(0)

        if init_h:
            self.h0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device, requires_grad=False)
            self.c0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device, requires_grad=False)
        else:
            self.h0.data.zero_()
            self.c0.data.zero_()

        self.h0[-1] = self.latent_hidden_linear(z)

        generations = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long, device=device)
        input_sequence = torch.tensor([sos_idx] * batch_size, dtype=torch.long, device=device)

        hidden = (self.h0, self.c0)
        for i in range(self.max_sequence_length):
            input_embedding = F.dropout(self.dec_word_vecs(input_sequence).view(batch_size, 1, self.embedding_size), training=training)
            dec_input = torch.cat([input_embedding, z.view(batch_size, 1, self.latent_dim)], dim=2)  # TODO: project z to embedding space before concat?
            output, hidden = self.dec_rnn(dec_input, hidden)
            dec_linear_input = output.contiguous()
            dec_linear_input = F.dropout(dec_linear_input, training=training)  # TODO: this dropout is necessary?
            preds = self.dec_linear(dec_linear_input.view(batch_size, self.dec_h_dim))
            probs = F.softmax(preds, dim=1)
            samples = torch.multinomial(probs, 1)
            generations[:, i] = samples.view(-1).data
            input_sequence = samples.view(-1)

        return generations


##--------------------------------------------------------------------------------------------------------------------##

def train_grid(args_job, output_dir_job, output_dir, return_dict):
    logger = setup_logging('main', output_dir, console=True)

    args = parse_args()
    args = pygrid.overwrite_opt(args, args_job)

    set_seed(args.seed)
    set_gpu(args.device, deterministic=True)

    logger.info(args)

    val_nll, val_rec_abp, val_kl_abp, val_ppl_bound_abp, test_nll, test_rec_abp, test_kl_abp, test_ppl_bound_abp, cls_loss, cls_acc, cls_loss_2, cls_acc_2 = train(args, output_dir, logger)

    return_dict['stats'] = {'val_nll':val_nll, 'val_rec_abp':val_rec_abp, 'val_kl_abp':val_kl_abp, 'val_ppl_bound_abp':val_ppl_bound_abp, 'test_nll':test_nll, 'test_rec_abp':test_rec_abp, 'test_kl_abp':test_kl_abp, 'test_ppl_bound_abp':test_ppl_bound_abp, 'cls_loss':cls_loss, 'cls_acc':cls_acc, 'cls_loss_2':cls_loss_2, 'cls_acc_2':cls_acc_2}

    logger.info('done')


def train(args, output_dir, logger):

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')

    args.cls_save_path = '{}/cls'.format(output_dir)

    ############### data ###############

    if args.dataset == 'yelp':

        from labeled_data import MonoTextData, VocabEntry

        vocab = {}
        with open(args.vocab_file) as fvocab:
            for i, line in enumerate(fvocab):
                vocab[line.strip()] = i

        vocab = VocabEntry(vocab)
        vocab_size = len(vocab)

        train_data_all = MonoTextData(args.train_data, label=args.label, vocab=vocab)
        val_data_all = MonoTextData(args.val_data, label=args.label, vocab=vocab)
        test_data_all = MonoTextData(args.test_data, label=args.label, vocab=vocab)

        train_data, train_labels = train_data_all.create_data_batch_labels(batch_size=args.batch_size, device=device, batch_first=True)
        val_data, val_labels = val_data_all.create_data_batch_labels(batch_size=32, device=device, batch_first=True)
        test_data, test_labels = test_data_all.create_data_batch_labels(batch_size=32, device=device, batch_first=True)

        cls_train_data_all = MonoTextData(args.cls_train_data, label=args.label, vocab=vocab)
        cls_train_data, cls_train_labels = cls_train_data_all.create_data_batch_labels(batch_size=args.batch_size, device=device, batch_first=True)

        def get_batch(data, i):
            sents = data[i]
            batch_size, sent_len = sents.size()
            length = sent_len - 1
            return sents, length, batch_size

    else:

        train_data = Dataset(args.train_file)
        val_data = Dataset(args.val_file)
        test_data = Dataset(args.test_file)

        vocab_size = int(train_data.vocab_size)

        indexer = Indexer()
        indexer.load_vocab(args.vocab_file)

        def get_batch(data, i):
            sents, length, batch_size = data[i]
            return sents, length, batch_size

    logger.info('Train data: %d batches' % len(train_data))
    logger.info('Val data: %d batches' % len(val_data))
    logger.info('Test data: %d batches' % len(test_data))
    logger.info('Word vocab size: %d' % vocab_size)




    ############### model ###############

    checkpoint_dir = output_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    suffix = "%s_%s.pt" % (args.model, 'bl')
    checkpoint_path = os.path.join(checkpoint_dir, suffix)

    if args.train_from == '':
        model = RNNVAE(args, vocab_size=vocab_size,
                       enc_word_dim=args.enc_word_dim,
                       enc_h_dim=args.enc_h_dim,
                       enc_num_layers=args.enc_num_layers,
                       dec_word_dim=args.dec_word_dim,
                       dec_h_dim=args.dec_h_dim,
                       dec_num_layers=args.dec_num_layers,
                       dec_dropout=args.dec_dropout,
                       latent_dim=args.latent_dim,
                       mode=args.model)
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
    else:
        logger.info('loading model from ' + args.train_from)
        checkpoint = torch.load(args.train_from)
        model = checkpoint['model']

    logger.info("model architecture")
    logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.warmup == 0:
        args.beta = 1.
    else:
        args.beta = 0.1

    criterion = nn.NLLLoss(ignore_index=0)
    model.cuda()
    criterion.cuda()
    model.train()

    def variational_loss(input, sents, model, z=None):
        mean, logvar = input
        z_samples = model._reparameterize(mean, logvar, z)
        preds = model._dec_forward(sents, z_samples)
        nll = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(preds.size(1))])
        kl = utils.kl_loss_diag(mean, logvar)
        return nll + args.beta * kl

    update_params = list(model.dec.parameters())
    meta_optimizer = OptimN2N(variational_loss, model, update_params, eps=args.eps,
                              lr=[args.svi_lr1, args.svi_lr2],
                              iters=args.svi_steps, momentum=args.momentum,
                              acc_param_grads=args.train_n2n == 1,
                              max_grad_norm=args.svi_max_grad_norm)


    ############### eval with cache ###############

    if args.use_cache:
        train_data_batch_cache = torch.load('cache/cache_100_labels/train_data_batch_cache_1')
        train_labels_batch_cache = torch.load('cache/cache_100_labels/train_labels_batch_cache_1')
        test_data_batch_cache = torch.load('cache/cache_100_labels/test_data_batch_cache_1')
        test_labels_batch_cache = torch.load('cache/cache_100_labels/test_labels_batch_cache_1')
        val_data_batch_cache = torch.load('cache/cache_100_labels/val_data_batch_cache_1')
        val_labels_batch_cache = torch.load('cache/cache_100_labels/val_labels_batch_cache_1')


        cls_loss, cls_acc = eval_cls(args, logger, model, device, train_data_batch_cache, train_labels_batch_cache, test_data_batch_cache, test_labels_batch_cache, val_data_batch_cache, val_labels_batch_cache)

        cls_loss_2 = 0.
        cls_acc_2 = 0.
        val_nll = 0.
        val_rec_abp = 0.
        val_kl_abp = 0.
        val_ppl_bound_abp = 0.
        test_nll = 0.
        test_rec_abp = 0.
        test_kl_abp = 0.
        test_ppl_bound_abp = 0.
        return val_nll, val_rec_abp, val_kl_abp, val_ppl_bound_abp, test_nll, test_rec_abp, test_kl_abp, test_ppl_bound_abp, cls_loss, cls_acc, cls_loss_2, cls_acc_2


    ############### train ###############

    t = 0
    best_val_nll = 1e5
    best_epoch = 0
    val_stats = []
    epoch = 0
    compute_kl = 0
    z_means = torch.zeros(5, args.latent_dim, device=device, dtype=torch.float)
    while epoch < args.num_epochs:
        start_time = time.time()
        epoch += 1
        logger.info('Starting epoch %d' % epoch)
        train_nll_abp = 0.
        train_kl_abp = 0.
        num_sents = 0
        num_words = 0
        b = 0

        for i in np.random.permutation(len(train_data)):

            if args.warmup > 0:
                args.beta = min(1., args.beta + 1. / (args.warmup * len(train_data)))

            sents, length, batch_size = get_batch(train_data, i)
            if args.device >= 0:
                sents = sents.cuda()
            b += 1

            optimizer.zero_grad()
            z_0 = sample_p_0(args, sents)
            in_mask, out_mask = dropout_mask(sents, args.dec_word_dim, args.dec_h_dim, p=0.5)
            z_samples, z_grads = model.infer_z(z_0, sents, args.beta, in_mask=in_mask, out_mask=out_mask)
            preds = model._dec_forward(sents, z_samples, in_mask=in_mask, out_mask=out_mask)
            nll_abp = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(length)])
            train_nll_abp += nll_abp.item() * batch_size
            abp_loss = nll_abp
            abp_loss.backward(retain_graph=True)

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()
            num_sents += batch_size
            num_words += batch_size * length

            if b % args.sample_every == 0:

                z_samples_container = []
                for z_i in range(args.num_z_samples):
                    z_0 = sample_p_0(args, sents)
                    z_samples, _ = model.infer_z(z_0, sents)  # TODO change mask
                    z_samples_container.append(z_samples)
                    z_means = torch.stack(z_samples_container, dim=2).mean(dim=2)

            if b % args.print_every == 0:
                with torch.no_grad():
                    z_var = ' '.join(['{:10.6f}'.format(_z_var) for _z_var in z_means.std(dim=0).pow(2)])
                    param_norm = sum([p.norm() ** 2 for p in model.parameters()]).item() ** 0.5
                    z_grad_t_str = ' '.join(['{:8.2f}'.format(g) for g in z_grads])
                    z_norm_str = '[{:8.2f} {:8.2f}]'.format(torch.norm(z_0, dim=0).mean(), torch.norm(z_samples, dim=0).mean())
                    z_disp_str = '{:8.2f}'.format(torch.norm(z_0 - z_samples, dim=0).mean())

                logger.info('Iters={}, Epoch={:4d}, Batch={:4d}/{:4d}, LR={:8.6f}, TrainABP_NLL={:10.4f}, TrainABP_REC={:10.4f}, '
                            'TrainABP_KL={:10.4f}, TrainABP_PPL={:10.4f}, |Param|={:10.4f}, z_norm_str={}, z_grad_t_str={}, z_disp_str={}, z_var={}, '
                            'BestValPerf={:10.4f}, BestEpoch={:4d}, Beta={:10.4f}'.format(
                    t, epoch, b + 1, len(train_data), args.lr, (train_nll_abp + train_kl_abp) / num_sents,
                              train_nll_abp / num_sents, train_kl_abp / num_sents,
                    np.exp((train_nll_abp + train_kl_abp) / num_words),
                    param_norm, z_norm_str, z_grad_t_str, z_disp_str, z_var, best_val_nll, best_epoch, args.beta))


        epoch_train_time = time.time() - start_time
        logger.info('Time Elapsed: %.1fs' % epoch_train_time)


    ############### eval ###############

    compute_kl = 0
    val_kl_abp = 0.
    val_ppl_bound_abp = 0.
    test_kl_abp = 0.
    test_ppl_bound_abp = 0.
    val_nll, val_rec_abp  = eval(args, logger, val_data, get_batch, model, device_cpu, device, compute_kl, meta_optimizer)
    test_nll, test_rec_abp = eval(args, logger, test_data, get_batch, model, device_cpu, device, compute_kl, meta_optimizer)

    train_data_batch_cache, train_labels_batch_cache = create_cls_data(args, model, cls_train_data, cls_train_labels, args.M, args.latent_dim, device, training=True)
    val_data_batch_cache, val_labels_batch_cache = create_cls_data(args, model, val_data, val_labels, args.M, args.latent_dim, device, training=True)
    test_data_batch_cache, test_labels_batch_cache = create_cls_data(args, model, test_data, test_labels, args.M, args.latent_dim, device, training=True)

    files_to_save = [train_data_batch_cache, train_labels_batch_cache, val_data_batch_cache,
                     val_labels_batch_cache, test_data_batch_cache, test_labels_batch_cache]
    files_names_to_save = ['train_data_batch_cache', 'train_labels_batch_cache', 'val_data_batch_cache',
                           'val_labels_batch_cache', 'test_data_batch_cache', 'test_labels_batch_cache']
    for file, file_name in zip(files_to_save, files_names_to_save):
        torch.save(file, os.path.join(output_dir, file_name + '_1'))

    cls_loss, cls_acc = eval_cls(args, logger, model, device, train_data_batch_cache, train_labels_batch_cache, test_data_batch_cache, test_labels_batch_cache, val_data_batch_cache, val_labels_batch_cache)

    cls_loss_2 = 0.
    cls_acc_2 = 0.
    return val_nll, val_rec_abp, val_kl_abp, val_ppl_bound_abp, test_nll, test_rec_abp, test_kl_abp, test_ppl_bound_abp, cls_loss, cls_acc, cls_loss_2, cls_acc_2


##--------------------------------------------------------------------------------------------------------------------##

class LinearDiscriminator(nn.Module):
    """docstring for LinearDiscriminator"""

    def __init__(self, args):
        super(LinearDiscriminator, self).__init__()
        self.args = args

        self.linear = nn.Linear(args.latent_dim, args.ncluster)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def get_performance(self, batch_data, batch_labels):
        logits = self.linear(batch_data)
        loss = self.loss(logits, batch_labels)

        _, pred = torch.max(logits, dim=1)
        correct = torch.eq(pred, batch_labels).float().sum().item()

        return loss, correct


def create_cls_data(args, model, train_data_batch, train_labels_batch, M, latent_dim, device, training=True):
    cls_train_batcth_labels_list = []
    cls_train_batch_size_list = []
    cls_train_sent_len_list = []
    data_size = sum([len(d) for d in train_labels_batch])
    cls_train_batch_data_tensor = torch.zeros((data_size, latent_dim, M), dtype=torch.float, device=device)
    for rept in range(M):

        cls_train_batch_data_list = []
        for i in range(len(train_data_batch)):
            batch_data = train_data_batch[i]
            batch_labels = train_labels_batch[i]
            batch_size, sent_len = batch_data.size()

            z_0 = sample_p_0(args, batch_data)
            z_samples = model.infer_z(z_0, batch_data, training=training)[0].detach().clone()
            cls_train_batch_data_list.append(z_samples)

            if rept == 0:
                cls_train_batcth_labels_list.append(batch_labels)
                cls_train_sent_len_list.append(sent_len)
                cls_train_batch_size_list.append(batch_size)
        cls_train_batch_data_list = torch.cat(cls_train_batch_data_list, dim=0)
        cls_train_batch_data_tensor[:, :, rept] = cls_train_batch_data_list
    cls_train_batch_data_tensor = cls_train_batch_data_tensor.mean(dim=-1)

    cls_train_batch_data_list = []
    acc_batch_size = 0
    for i in range(len(cls_train_batch_size_list)):
        batch_size = cls_train_batch_size_list[i]
        batch_data = cls_train_batch_data_tensor[acc_batch_size:acc_batch_size + batch_size, :]
        cls_train_batch_data_list.append(batch_data)
        acc_batch_size += batch_size

    return cls_train_batch_data_list, cls_train_batcth_labels_list


def eval_cls(args, logger, model, device, train_data_batch, train_labels_batch, test_data_batch, test_labels_batch, val_data_batch, val_labels_batch):

    discriminator = LinearDiscriminator(args).to(device)
    discriminator.train()

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.cls_lr, momentum=args.cls_momentum)
    opt_dict['lr'] = args.cls_lr

    clip_grad = 5.0
    decay_epoch = 2
    lr_decay = 0.5
    max_decay = 5
    log_niter = 100

    start = time.time()
    best_loss = 1e4

    iter_ = 0
    decay_cnt = 0
    acc_cnt = 1
    acc_loss = 0.
    for epoch in range(args.cls_epochs):
        report_loss = 0
        report_correct = report_num_words = report_num_sents = 0
        acc_batch_size = 0
        optimizer.zero_grad()
        for i in np.random.permutation(len(train_data_batch)):

            batch_data = train_data_batch[i].to(device)
            batch_labels = train_labels_batch[i]
            batch_labels = [int(x) for x in batch_labels]
            batch_labels = torch.tensor(batch_labels, dtype=torch.long, requires_grad=False, device=device)

            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size
            report_num_sents += batch_size
            acc_batch_size += batch_size

            # (batch_size)
            loss, correct = discriminator.get_performance(batch_data, batch_labels)

            acc_loss = acc_loss + loss.sum()

            if acc_cnt % args.cls_update_every == 0:
                acc_loss = acc_loss / acc_batch_size
                acc_loss.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)

                optimizer.step()
                optimizer.zero_grad()

                acc_cnt = 0
                acc_loss = 0
                acc_batch_size = 0

            acc_cnt += 1
            report_loss += loss.sum().item()
            report_correct += correct

            if iter_ % log_niter == 0:
                # train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
                train_loss = report_loss / report_num_sents

                logger.info('epoch: %d, iter: %d, avg_loss: %.4f, acc %.4f, time %.2fs' %
                        (epoch, iter_, train_loss, report_correct / report_num_sents,
                         time.time() - start))

                # sys.stdout.flush()

            iter_ += 1

        logger.info('lr {}'.format(opt_dict["lr"]))

        loss, acc = test(logger, device, discriminator, val_data_batch, val_labels_batch, "VAL", args)

        if loss < best_loss:
            logger.info('update best loss')
            best_loss = loss
            best_acc = acc
            torch.save(discriminator.state_dict(), args.cls_save_path)

        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >= args.cls_load_best_epoch:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                discriminator.load_state_dict(torch.load(args.cls_save_path))
                logger.info('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                optimizer = torch.optim.SGD(discriminator.parameters(), lr=opt_dict["lr"], momentum=args.cls_momentum)
                opt_dict['lr'] = opt_dict["lr"]

        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.cls_test_nepoch == 0:
            # with torch.no_grad():
            loss, acc = test(logger, device, discriminator, test_data_batch, test_labels_batch, "TEST", args)

    # compute importance weighted estimate of log p(x)
    discriminator.load_state_dict(torch.load(args.cls_save_path))
    cls_loss, cls_acc = test(logger, device, discriminator, test_data_batch, test_labels_batch, "TEST", args)

    return cls_loss, cls_acc


def test(logger, device, model, test_data_batch, test_labels_batch, mode, args, verbose=True):

    report_correct = report_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_labels = test_labels_batch[i]
        batch_labels = [int(x) for x in batch_labels]

        batch_labels = torch.tensor(batch_labels, dtype=torch.long, requires_grad=False, device=device)

        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size
        loss, correct = model.get_performance(batch_data, batch_labels)

        loss = loss.sum()

        report_loss += loss.item()
        report_correct += correct

    test_loss = report_loss / report_num_sents
    acc = report_correct / report_num_sents

    if verbose:
        logger.info('%s --- avg_loss: %.4f, acc: %.4f' % (mode, test_loss, acc))

    return test_loss, acc

##--------------------------------------------------------------------------------------------------------------------##

def eval(args, logger, data, get_batch, model, device_cpu, device_gpu, compute_kl, meta_optimizer):
    # model.eval()
    criterion = nn.NLLLoss().to(device_gpu)
    num_sents = 0
    num_words = 0
    total_nll_abp = 0.
    total_kl_abp = 0.
    for i in range(len(data)):
        sents, length, batch_size = get_batch(data, i)
        num_words += batch_size * length
        num_sents += batch_size
        sents = sents.to(device_gpu)

        z_0 = sample_p_0(args, sents).to(device_gpu)
        model = model.to(device_gpu)
        z_samples = model.infer_z(z_0, sents, training=False)[0]
        preds = model._dec_forward(sents, z_samples, training=False)
        nll_abp = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(length)])
        total_nll_abp += nll_abp.item() * batch_size

    # assert total_kl_abp == 0.
    nll_abp = (total_nll_abp + total_kl_abp) / num_sents
    rec_abp = total_nll_abp / num_sents

    logger.info('ABP NLL: %.4f, ABP REC: %.4f' % (nll_abp, rec_abp))
    model.train()

    return nll_abp, rec_abp


##--------------------------------------------------------------------------------------------------------------------##

def sample_p_0(args, x):
    return torch.randn(*[x.size(0), args.latent_dim], device=x.device)


def jacobian(inputs, outputs):
    return torch.stack(
        [torch.autograd.grad(outputs[:, i].sum(), inputs, retain_graph=True, create_graph=True)[0] for i in
         range(outputs.size(1))], dim=-1)




##--------------------------------------------------------------------------------------------------------------------##

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


def merge_dicts(a, b, c):
    d = {}
    d.update(a)
    d.update(b)
    d.update(c)
    return d


def main():

    use_pygrid = True

    if use_pygrid:

        # TODO enumerate gpu devices here
        device_ids = [1]
        workers = len(device_ids)

        # set devices
        pygrid.init_mp()
        pygrid.fill_queue(device_ids)

        fs_suffix = './'

        # set opts
        get_opts_filename = lambda exp: fs_suffix + '{}.csv'.format(exp)
        exp_id = pygrid.get_exp_id(__file__)

        write_opts = lambda opts: pygrid.write_opts(opts, lambda: open(get_opts_filename(exp_id), mode='w'))
        read_opts = lambda: pygrid.read_opts(lambda: open(get_opts_filename(exp_id), mode='r'))

        output_dir = fs_suffix + pygrid.get_output_dir(exp_id)
        os.makedirs(output_dir + '/samples')

        if not os.path.exists(get_opts_filename(exp_id)):
            write_opts(create_args_grid())
        write_opts(pygrid.reset_job_status(read_opts()))

        # set logging
        logger = pygrid.setup_logging('main', output_dir, console=True)
        logger.info('available devices {}'.format(device_ids))

        # run
        copy_source(__file__, output_dir)
        pygrid.run_jobs(logger, exp_id, output_dir, workers, train_grid, read_opts, write_opts, update_job_result)
        logger.info('done')

    else:

        # preamble
        exp_id = pygrid.get_exp_id(__file__)
        fs_suffix = './'
        output_dir = fs_suffix + pygrid.get_output_dir(exp_id)

        # run
        copy_source(__file__, output_dir)
        # opt = create_opts()[0]
        opt = {'job_id': int(0), 'status': 'open', 'device': 0}
        train(opt, output_dir, output_dir, {})


if __name__ == '__main__':
    main()
