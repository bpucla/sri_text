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
import logger


from preprocess_text import Indexer

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_file', default='datasets/ptb/ptb-train.hdf5')
parser.add_argument('--val_file', default='datasets/ptb/ptb-val.hdf5')
parser.add_argument('--test_file', default='datasets/ptb/ptb-test.hdf5')
parser.add_argument('--vocab_file', default='datasets/ptb/ptb.dict')
parser.add_argument('--train_from', default='cache/ckpt/model_ckpt.pt')
parser.add_argument('--eval_only', default=True, type=bool)

# SRI options
parser.add_argument('--z_n_iters', type=int, default=20)
parser.add_argument('--z_step_size', type=float, default=0.2)
parser.add_argument('--z_with_noise', type=int, default=1)
parser.add_argument('--prior_sigma', type=float, default=1.0)
parser.add_argument('--num_z_samples', type=int, default=10)
parser.add_argument('--nll_M', default=10, type=int)



# Model options
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--enc_word_dim', default=256, type=int)
parser.add_argument('--enc_h_dim', default=256, type=int)
parser.add_argument('--enc_num_layers', default=1, type=int)
parser.add_argument('--dec_word_dim', default=256, type=int)
parser.add_argument('--dec_h_dim', default=256, type=int)
parser.add_argument('--dec_num_layers', default=1, type=int)
parser.add_argument('--dec_dropout', default=0.5, type=float)
parser.add_argument('--model', default='abp', type=str, choices = ['abp', 'vae'])
parser.add_argument('--train_n2n', default=1, type=int)
parser.add_argument('--train_kl', default=1, type=int)

# Optimization options
parser.add_argument('--checkpoint_dir', default='models/ptb')
parser.add_argument('--slurm', default=0, type=int)
parser.add_argument('--warmup', default=0, type=int)
parser.add_argument('--num_epochs', default=60, type=int)
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
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--sample_every', type=int, default=1000)
parser.add_argument('--kl_every', type=int, default=100)
parser.add_argument('--compute_kl', type=int, default=1)
parser.add_argument('--test', type=int, default=0)


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
    self.vocab_size = vocab_size
    self.step_sizes = [0.8, 0.7, 0.6, 0.5, 0.2, 0.2, 0.1, 0.1, 0.05] + [0.05] * 11

    if mode == 'savae' or mode == 'vae':
      self.enc_word_vecs = nn.Embedding(vocab_size, enc_word_dim)
      self.latent_linear_mean = nn.Linear(enc_h_dim, latent_dim)
      self.latent_linear_logvar = nn.Linear(enc_h_dim, latent_dim)
      self.enc_rnn = nn.LSTM(enc_word_dim, enc_h_dim, num_layers=enc_num_layers,
                             batch_first=True)
      self.enc = nn.ModuleList([self.enc_word_vecs, self.enc_rnn,
                                self.latent_linear_mean, self.latent_linear_logvar])
    elif mode == 'autoreg':
      latent_dim = 0

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

  def infer_z(self, z, sent, beta=1., step_size=0.8, training=True, T=0.2):
    args = self.args
    target = sent.detach().clone()
    target = target[:, 1:]
    z_grads_norm = []
    for i in range(args.z_n_iters):
      # z = torch.autograd.Variable(z.detach().clone(), requires_grad=True)
      z = z.detach().clone()
      z.requires_grad = True
      assert z.grad is None
      logp = self._dec_forward(sent, z, training=training)  # TODO: turn off dropout in inference?
      logp = logp.view(-1, logp.size(2))
      nll = F.nll_loss(logp, target.reshape(-1), reduction='sum', ignore_index=0)  # TODO remove hard-coding
      nll.backward()
      z_grad = z.grad.detach().clone()
      z = z - 0.5 * self.step_sizes[i] * self.step_sizes[i] * (beta*z + z.grad)
      if args.z_with_noise:
        z += T * self.step_sizes[i] * torch.randn_like(z)

      z_grads_norm.append(torch.norm(z_grad, dim=0).mean().cpu().numpy())

    z = z.detach()

    return z, z_grads_norm

  def infer_z_grad(self, z, sent, beta=1., step_size=0.8, training=True, T=0.2):
    args = self.args
    target = sent.detach().clone()
    target = target[:, 1:]
    for i in range(args.z_n_iters):
      logp = self._dec_forward(sent, z, training=training)
      logp = logp.view(-1, logp.size(2))
      nll = F.nll_loss(logp, target.reshape(-1), reduction='sum', ignore_index=0)  # TODO remove hard-coding
      z_grad = torch.autograd.grad(nll, z, retain_graph=True, create_graph=True)[0]
      z = z - 0.5 * self.step_sizes[i] * self.step_sizes[i] * (beta * z + z_grad)
      if args.z_with_noise:
        z += T * self.step_sizes[i] * torch.randn_like(z)
    return z


  def kl_single(self, z, sent, T, device, beta=1., step_size=0.8, training=True):
    z = z.detach().clone().requires_grad_(True)
    z_k = self.infer_z_grad(z, sent, beta, step_size, training, T)
    J = jacobian(z, z_k).squeeze()
    sum_log_abs_det_jacobians = torch.slogdet(J)[1]
    if torch.isinf(sum_log_abs_det_jacobians):
        logger.info('inf')
    prior = torch.distributions.MultivariateNormal(torch.zeros(z.size(-1)).to(device), torch.eye(z.size(-1)).to(device))
    log_p_z_0 = prior.log_prob(z.squeeze())
    log_p_z_k = prior.log_prob(z_k.squeeze())
    kl = log_p_z_0 - sum_log_abs_det_jacobians - log_p_z_k

    return kl

  def compute_nll_single(self, z, sent, T, device, beta=1., step_size=0.8, training=True):
    with torch.backends.cudnn.flags(enabled=False):
      z = z.detach().clone().requires_grad_(True)
      z_k = self.infer_z_grad(z, sent, beta, step_size, training, T)
      J = jacobian(z, z_k).squeeze()
      sum_log_abs_det_jacobians = torch.slogdet(J)[1]
      if torch.isinf(sum_log_abs_det_jacobians):
          logger.info('inf')
      prior = torch.distributions.MultivariateNormal(torch.zeros(z.size(-1)).to(device), torch.eye(z.size(-1)).to(device))
      log_p_z_0 = prior.log_prob(z.squeeze())
      log_p_z_k = prior.log_prob(z_k.squeeze())
      kl = log_p_z_k - log_p_z_0 + sum_log_abs_det_jacobians

      return kl, z_k

  def compute_nll_batch(self, z, sent, device, beta=1., step_size=0.8, training=True):
    with torch.backends.cudnn.flags(enabled=False):
      # z = z.detach().clone().requires_grad_(True)
      # z_k = self.infer_z_grad(z, sent, beta, step_size, training)
      # J = jacobian(z, z_k).squeeze()

      def compute_jacobian_batch(z, sent, device, beta=beta, step_size=step_size, training=training):
        # z = z.detach().clone().requires_grad_(True)
        z_k = self.infer_z_grad(z, sent, beta, step_size, training)
        return z_k

      
      # J = self.compute_jacobian_batch(z, sent, device, beta=beta, step_size=step_size, training=training)
      
      J = torch.autograd.functional.jacobian(lambda _z: compute_jacobian_batch(_z, sent, device, beta=beta, step_size=step_size, training=training), z)
      batch_size = z.size(0)
      J = J[range(batch_size), :, range(batch_size), :]

      sum_log_abs_det_jacobians = torch.slogdet(J)[1]
      z = z.detach().clone().requires_grad_(True)
      z_k = self.infer_z(z, sent, beta, step_size, training)[0]
      if any(torch.isinf(sum_log_abs_det_jacobians)):
          logger.info('inf')
      prior = torch.distributions.MultivariateNormal(torch.zeros(z.size(-1)).to(device), torch.eye(z.size(-1)).to(device))
      log_p_z_0 = prior.log_prob(z.squeeze())
      log_p_z_k = prior.log_prob(z_k.squeeze())
      kl = log_p_z_k - log_p_z_0 + sum_log_abs_det_jacobians

      return kl, z_k

  def _dec_forward(self, sent, q_z, init_h=True, training=True):
    self.word_vecs = F.dropout(self.dec_word_vecs(sent[:, :-1]), training=training)
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
    dec_linear_input = F.dropout(dec_linear_input, training=training)
    preds = self.dec_linear(dec_linear_input.view(
      self.word_vecs.size(0) * self.word_vecs.size(1), -1)).view(
      self.word_vecs.size(0), self.word_vecs.size(1), -1)
    return preds

  def inference(self, device, sos_idx, max_len=None, n=4, z=None, init_h=True, training=True):

    batch_size = z.size(0)

    if init_h:
      self.h0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device, requires_grad=False)
      self.c0 = torch.zeros((self.dec_num_layers, batch_size, self.dec_h_dim), dtype=torch.float, device=device, requires_grad=False)
    else:
      self.h0.data.zero_()
      self.c0.data.zero_()

    self.h0[-1] = self.latent_hidden_linear(z)

    if max_len is None:
      max_len = self.max_sequence_length
    generations = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    preds_sequence = torch.zeros(batch_size, max_len, self.vocab_size, dtype=torch.float, device=device)
    input_sequence = torch.tensor([sos_idx]*batch_size, dtype=torch.long, device=device)

    hidden = (self.h0, self.c0)
    for i in range(max_len):
      input_embedding = F.dropout(self.dec_word_vecs(input_sequence).view(batch_size, 1, self.embedding_size), training=training)
      dec_input = torch.cat([input_embedding, z.view(batch_size, 1, self.latent_dim)], dim=2)  #TODO: project z to embedding space before concat?
      output, hidden = self.dec_rnn(dec_input, hidden)
      dec_linear_input = output.contiguous()
      dec_linear_input = F.dropout(dec_linear_input, training=training) #TODO: this dropout is necessary?
      preds = self.dec_linear(dec_linear_input.view(batch_size, self.dec_h_dim))
      probs = F.softmax(preds, dim=1)
      samples = torch.multinomial(probs, 1)
      generations[:, i] = samples.view(-1).data
      preds_sequence[:, i, :] = preds
      input_sequence = samples.view(-1)

    return generations, preds_sequence




##--------------------------------------------------------------------------------------------------------------------##

def set_seed(seed):
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args, output_dir):
  set_seed(args.seed)

  train_data = Dataset(args.train_file)
  val_data = Dataset(args.val_file)
  test_data = Dataset(args.test_file)
  train_sents = train_data.batch_size.sum()
  vocab_size = int(train_data.vocab_size)
  logger.info('Train data: %d batches' % len(train_data))
  logger.info('Val data: %d batches' % len(val_data))
  logger.info('Test data: %d batches' % len(test_data))
  logger.info('Word vocab size: %d' % vocab_size)

  checkpoint_dir = output_dir
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  suffix = "%s_%s.pt" % (args.model, 'bl')
  checkpoint_path = os.path.join(checkpoint_dir, suffix)


  indexer = Indexer()
  indexer.load_vocab(args.vocab_file)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if args.train_from == '':
    model = RNNVAE(args, vocab_size = vocab_size,
                   enc_word_dim = args.enc_word_dim,
                   enc_h_dim = args.enc_h_dim,
                   enc_num_layers = args.enc_num_layers,
                   dec_word_dim = args.dec_word_dim,
                   dec_h_dim = args.dec_h_dim,
                   dec_num_layers = args.dec_num_layers,
                   dec_dropout = args.dec_dropout,
                   latent_dim = args.latent_dim,
                   mode = args.model)
    for param in model.parameters():
      param.data.uniform_(-0.1, 0.1)
  else:
    logger.info('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']

  logger.info("model architecture")
  print(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  if args.warmup == 0:
    args.beta = 1.
  else:
    args.beta = 0.1

  criterion = nn.NLLLoss(ignore_index=0)
  model.cuda()
  criterion.cuda()
  model.train()

  def variational_loss(input, sents, model, z = None):
    mean, logvar = input
    z_samples = model._reparameterize(mean, logvar, z)
    preds = model._dec_forward(sents, z_samples)
    nll = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(preds.size(1))])
    kl = utils.kl_loss_diag(mean, logvar)
    return nll + args.beta*kl

  update_params = list(model.dec.parameters())
  meta_optimizer = OptimN2N(variational_loss, model, update_params, eps = args.eps, 
                            lr = [args.svi_lr1, args.svi_lr2],
                            iters = args.svi_steps, momentum = args.momentum,
                            acc_param_grads= args.train_n2n == 1,
                            max_grad_norm = args.svi_max_grad_norm)
  # if args.test == 1:
  #   args.beta = 1
  #   test_data = Dataset(args.test_file)
  #   eval(test_data, model, meta_optimizer)
  #   exit()

  t = 0
  best_val_nll = 1e5
  best_epoch = 0
  val_stats = []
  epoch = 0
  compute_kl = 0
  z_means = torch.zeros(5, args.latent_dim, device=device, dtype=torch.float)


  train_recons_batch = train_data[100][0][:10, :].to(device)
  test_recons_batch = test_data[100][0][:10, :].to(device)


  if args.eval_only:
    test_nll = eval_multi_batch(logger, args, test_data, model, device, device, 1, args.nll_M, meta_optimizer)
    exit()

  T = 0.05

  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1
    logger.info('Starting epoch %d' % epoch)
    train_nll_abp = 0.
    train_kl_abp = 0.
    num_sents = 0
    num_words = 0
    b = 0


    if epoch > 15:
      T = min(T+0.05, 0.2)
      logger.info(f'T={T}')


    checkpoint_path_epoch = f'{output_dir}/model_{epoch}.pt'
    model.cpu()
    checkpoint = {
      'args': args.__dict__,
      'model': model,
    }
    logger.info('Save checkpoint to %s' % checkpoint_path_epoch)
    torch.save(checkpoint, checkpoint_path_epoch)
    model.cuda()

    for i in np.random.permutation(len(train_data)):

      if args.warmup > 0:
        args.beta = min(1., args.beta + 1./(args.warmup*len(train_data)))

      sents, length, batch_size = train_data[i]
      if args.gpu >= 0:
        sents = sents.cuda()
      b += 1

      optimizer.zero_grad()
      z_0 = sample_p_0(sents)
      z_samples, z_grads = model.infer_z(z_0, sents, args.beta, args.z_step_size, T=T)
      preds = model._dec_forward(sents, z_samples)
      nll_abp = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      train_nll_abp += nll_abp.item()*batch_size
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
          z_0 = sample_p_0(sents)
          z_samples, _ = model.infer_z(z_0, sents, args.beta, args.z_step_size, T=T)
          z_samples_container.append(z_samples)
          z_means = torch.stack(z_samples_container, dim=2).mean(dim=2)



      if b % args.print_every == 0:
        with torch.no_grad():
          z_var = ' '.join(['{:10.6f}'.format(_z_var) for _z_var in z_means.std(dim=0).pow(2)])
          param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
          z_grad_t_str = ' '.join(['{:8.2f}'.format(g) for g in z_grads])
          z_norm_str = '[{:8.2f} {:8.2f}]'.format(torch.norm(z_0, dim=1).mean(), torch.norm(z_samples, dim=1).mean())
          z_disp_str = '{:8.2f}'.format(torch.norm(z_0 - z_samples, dim=1).mean())

        logger.info('Iters={}, Epoch={:4d}, Batch={:4d}/{:4d}, LR={:8.6f}, TrainABP_NLL={:10.4f}, TrainABP_REC={:10.4f}, '
                    'TrainABP_KL={:10.4f}, TrainABP_PPL={:10.4f}, |Param|={:10.4f}, z_norm_str={}, z_grad_t_str={}, z_disp_str={}, z_var={}, '
                    'BestValPerf={:10.4f}, BestEpoch={:4d}, Beta={:10.4f}'.format(
          t, epoch, b+1, len(train_data), args.lr, (train_nll_abp + train_kl_abp)/num_sents,
               train_nll_abp / num_sents, train_kl_abp / num_sents,
               np.exp((train_nll_abp + train_kl_abp)/num_words),
               param_norm, z_norm_str, z_grad_t_str, z_disp_str, z_var, best_val_nll, best_epoch, args.beta))

    epoch_train_time = time.time() - start_time
    logger.info('Time Elapsed: %.1fs' % epoch_train_time)

    logger.info('--------------------------------')
    logger.info('Checking test perf...')
    logger.record_tabular('Epoch', epoch)
    logger.record_tabular('Mode', 'Test')
    logger.record_tabular('LR', args.lr)
    logger.record_tabular('Epoch Train Time', epoch_train_time)
    compute_kl = 0
    test_nll = eval(logger, args, test_data, model, device, device, compute_kl, args.nll_M, meta_optimizer, T)
    compute_kl = 0


##--------------------------------------------------------------------------------------------------------------------##

##--------------------------------------------------------------------------------------------------------------------##

def eval(logger, args, data, model, device_cpu, device_gpu, compute_nll, M, meta_optimizer, T):
  import random

  criterion = nn.NLLLoss().cuda()
  num_sents = 0
  num_words = 0
  total_nll_abp = 0.
  total_kl_abp = 0.
  nll = 0.
  batch_len = 0
  kl = 0.
  N = 10
  while batch_len < (N+1):
    batch_id = random.randint(2, len(data)-1)
    batch_len = data[batch_id][2]
  num_words_ppl = data[batch_id][1]

  for l in range(len(data)):
    sents, length, batch_size = data[l]
    num_words += batch_size * length
    num_sents += batch_size
    if args.gpu >= 0:
      sents = sents.cuda()

    z_0 = sample_p_0(sents)
    z_samples = model.infer_z(z_0, sents, args.beta, args.z_step_size, training=False)[0]
    preds = model._dec_forward(sents, z_samples, training=False)
    nll_abp = sum([criterion(preds[:, l], sents[:, l + 1]) for l in range(length)])
    total_nll_abp += nll_abp.item() * batch_size


    if compute_nll and l == batch_id:
      logger.info('batch_id %d' % batch_id)
      sents = sents.detach().clone()
      negative_kls = torch.zeros(sents.size(0), M).to(device_cpu)
      nll_minus_conditional = torch.zeros(sents.size(0), M).to(device_cpu)
      model_cpu = model.to(device_cpu)
      sents_cpu = sents.to(device_cpu)
      targets = sents_cpu.detach().clone()
      targets = targets[:, 1:]
      for i in range(M):
        z_0_cpu = sample_p_0(sents_cpu)
        z_k_cpu = torch.zeros_like(z_0_cpu).requires_grad_(True)
        for j in range(sents.size(0)):
          sents_j = sents_cpu[j, :].unsqueeze(0)
          z_0_j = z_0_cpu[j, :].unsqueeze(0)
          negative_kl, z_k_j = model_cpu.compute_nll_single(z_0_j, sents_j, T, device_cpu, args.beta, args.z_step_size, training=False)
          negative_kls[j, i] = negative_kl.detach().clone()
          del negative_kl

          z_k_cpu[j, :] = z_k_j.detach().clone()
          del z_k_j

        logp = model_cpu._dec_forward(sents_cpu, z_k_cpu, training=False)
        logp = logp.view(-1, logp.size(2))
        nll_conditional = F.nll_loss(logp, targets.reshape(-1), reduction='none', ignore_index=0).view(sents.size(0), sents.size(1)-1).sum(-1)
        nll_minus_conditional[:, i] = negative_kls[:, i] - nll_conditional

      nll = - (nll_minus_conditional.logsumexp(dim=-1).mean().item() - torch.tensor(M, dtype=torch.float).log().item())
      kl = - negative_kls.mean().item()

      model = model_cpu.to(device_gpu)
      del model_cpu, sents_cpu


  # assert total_kl_abp == 0.
  nll_abp = nll
  rec_abp = total_nll_abp / num_sents
  kl_abp = kl
  ppl_bound_abp = np.exp(nll / num_words_ppl)

  logger.record_tabular('ABP NLL', nll_abp)
  logger.record_tabular('ABP REC', rec_abp)
  logger.record_tabular('ABP KL', kl_abp)
  logger.record_tabular('ABP PPL', ppl_bound_abp)
  logger.dump_tabular()
  logger.info('ABP NLL: %.4f, ABP REC: %.4f, ABP KL: %.4f, ABP PPL: %.4f' %
              (nll_abp, rec_abp, kl_abp, ppl_bound_abp))
  model.train()
  return ppl_bound_abp


def eval_multi_batch(logger, args, data, model, device_cpu, device_gpu, compute_nll, M, meta_optimizer, T=0.2):
  import random

  criterion = nn.NLLLoss().cuda()
  num_sents = 0
  num_words = 0
  total_nll_abp = 0.
  total_kl_abp = 0.
  nll = 0.
  batch_len = 0
  kl = 0.
  N = 10
  kls = []
  nlls = []
  batch_sizes = []
  batch_ids_list = random.sample(list(range(len(data))), k=len(data))
  logger.info('batch list : {}'.format(batch_ids_list))
  batch_ids = batch_ids_list[:80]
  num_words_ppl = 0
  num_ppl_batch = 0
  ppl_stats = []


  for l in range(len(data)):
    sents, length, batch_size = data[l]
    num_words += batch_size * length
    num_sents += batch_size
    if args.gpu >= 0:
      sents = sents.cuda()

    z_0 = sample_p_0(sents)
    z_samples = model.infer_z(z_0, sents, args.beta, args.z_step_size, training=False)[0]
    preds = model._dec_forward(sents, z_samples, training=False)
    nll_abp = sum([criterion(preds[:, _l], sents[:, _l + 1]) for _l in range(length)])
    total_nll_abp += nll_abp.item() * batch_size


    if compute_nll and l in batch_ids:
      logger.info('------> batch_id %d' % num_ppl_batch)
      num_ppl_batch += 1
      sents = sents.detach().clone()
      negative_kls = torch.zeros(sents.size(0), M).to(device_cpu)
      nll_minus_conditional = torch.zeros(sents.size(0), M).to(device_cpu)
      model_cpu = model.to(device_cpu)
      sents_cpu = sents.to(device_cpu)
      targets = sents_cpu.detach().clone()
      targets = targets[:, 1:]
      for i in range(M):
        z_0_cpu = sample_p_0(sents_cpu)
        negative_kl, z_k_j = model_cpu.compute_nll_batch(z_0_cpu, sents, device_cpu, args.beta, args.z_step_size, training=False)
        negative_kls[:, i] = negative_kl.detach().clone()
        del negative_kl
        z_k_cpu = z_k_j.detach().clone()
        del z_k_j

        logp = model_cpu._dec_forward(sents_cpu, z_k_cpu, training=False)
        logp = logp.view(-1, logp.size(2))
        nll_conditional = F.nll_loss(logp, targets.reshape(-1), reduction='none', ignore_index=0).view(sents.size(0), sents.size(1)-1).sum(-1)
        nll_minus_conditional[:, i] = negative_kls[:, i] - nll_conditional

      nll = - (nll_minus_conditional.logsumexp(dim=-1) - torch.tensor(M, dtype=torch.float).log()).sum().item()
      kl = - negative_kls.mean(dim=-1).sum().item()
      nlls.append(nll)
      kls.append(kl)
      batch_sizes.append(batch_size)
      curr_num_words = batch_size * length
      num_words_ppl += curr_num_words

      model = model_cpu.to(device_gpu)
      del model_cpu, sents_cpu

      ppl_stats.append([l, batch_size, length, nll, kl, kl/batch_size, np.exp(nll / curr_num_words), sum(kls) / sum(batch_sizes), np.exp(sum(nlls) / num_words_ppl)])
      ppl_stats_np = np.array(ppl_stats)
      with open(f'{output_dir}/ppl_stats_{l}.txt', 'wb') as f:
        np.save(f, ppl_stats_np)


  nll_abp = nll
  rec_abp = total_nll_abp / num_sents
  kl_abp = sum(kls) / sum(batch_sizes)
  ppl_bound_abp = np.exp(sum(nlls) / num_words_ppl)

  logger.record_tabular('ABP NLL', nll_abp)
  logger.record_tabular('ABP REC', rec_abp)
  logger.record_tabular('ABP KL', kl_abp)
  logger.record_tabular('ABP PPL', ppl_bound_abp)
  logger.dump_tabular()
  logger.info('ABP NLL: %.4f, ABP REC: %.4f, ABP KL: %.4f, ABP PPL: %.4f' %
              (nll_abp, rec_abp, kl_abp, ppl_bound_abp))
  model.train()
  return ppl_bound_abp


##--------------------------------------------------------------------------------------------------------------------##


def sample_p_0(x):
  return torch.randn(*[x.size(0), args.latent_dim], device=x.device)

def jacobian(inputs, outputs):
  return torch.stack(
    [torch.autograd.grad(outputs[:, i].sum(), inputs, retain_graph=True, create_graph=True)[0] for i in
    range(outputs.size(1))], dim=-1)


def idx2word(idx, i2w, ending_idx):
  sent_str = [str()] * len(idx)

  for i, sent in enumerate(idx):

    for word_id in sent:
      word_id = word_id.item()

      if word_id == ending_idx:
        break
      sent_str[i] += i2w[word_id] + " "

    sent_str[i] = sent_str[i].strip() + "\n"
  return sent_str



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

def set_gpu(gpu, deterministic=True):
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

  if torch.cuda.is_available():
    torch.cuda.set_device(0)

    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
  args = parser.parse_args()
  exp_id = get_exp_id(__file__)
  output_dir = get_output_dir(exp_id)
  copy_source(__file__, output_dir)

  set_gpu(args.gpu)

  with logger.session(dir=output_dir, format_strs=['stdout', 'csv', 'log']):
    main(args, output_dir)
