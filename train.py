"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

""" Training Config & Helper Classes  """

import os
import json
from typing import NamedTuple
from tqdm import tqdm
from torch.autograd import Variable
import torch
import torch.nn as nn

from utils import get_device

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 8
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    r: float = 1.125

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

def generator_loss(model, batch, global_step, optimizer, cross_ent, sent_cross_ent, writer=None, prefix='pretrain'): # make sure loss is tensor
    input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, _ = batch
    logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
    del input_ids, segment_ids, input_mask, masked_pos
    loss_lm = cross_ent(logits_lm.transpose(1, 2), masked_ids) # for masked LM
    loss_lm = (loss_lm*masked_weights.float()).mean()
    loss_sop = sent_cross_ent(logits_clsf, is_next) # for sentence classification

    if writer:
        writer.add_scalars(prefix+'/G',
                        {'loss_lm': loss_lm.item(),
                        'loss_sop': loss_sop.item(),
                        'loss_total': (loss_lm + loss_sop).item(),
                        'lr': optimizer.get_lr()[0],
                        },
                        global_step)
    return loss_lm + loss_sop, logits_lm, loss_sop


def discriminator_loss(generator, discriminator, batch, global_step, optimizer, cross_ent, sent_cross_ent, writer=None, prefix='pretrain'): # make sure loss is tensor
    # input_ids is the output of generator
    #   0           1           2           3
    masked_ids, segment_ids, input_mask, input_ids, _, _, is_next, original_ids = batch
    masked_label = (masked_ids.long() != original_ids)

    # replace the non masked generator token with the original token
    non_masked_label = (masked_ids == original_ids) 
    input_ids[non_masked_label] = original_ids[non_masked_label]

    is_replaced = Variable((input_ids.long() != original_ids.long()).float()).to(get_device())

    logits_lm, logits_clsf = discriminator(input_ids, segment_ids, input_mask)
    logits_lm = logits_lm.squeeze(-1)
    loss_lm = cross_ent(logits_lm, is_replaced) # for masked LM
    loss_lm = loss_lm.mean()
    loss_sop = sent_cross_ent(logits_clsf, is_next) # for sentence classification

    if writer:
        writer.add_scalars(prefix+'/D',
                        {'loss_lm': loss_lm.item(),
                        'loss_sop': loss_sop.item(),
                        'loss_total': (loss_lm + loss_sop).item(),
                        'lr': optimizer.get_lr()[0],
                        },
                        global_step)
        writer.add_scalars(prefix+'/stats',
                        {
                            'replaced': is_replaced.sum().item()/len(input_ids),
                            'masked': masked_label.sum().item()/len(input_ids),
                        },
                        global_step)
    return loss_lm + loss_sop, loss_lm, loss_sop


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, get_loss, model_file=None, pretrain_file=None, data_parallel=True):
        """ Train Loop """
        self.model.train() # train mode
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)', dynamic_ncols=True)
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss = get_loss(model, batch, global_step).mean() # mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step = global_step + 1
                loss_detached = loss.detach()
                del loss
                loss_sum = loss_sum + loss_detached
                iter_bar.set_description('Iter (loss=%5.3f)'%loss_detached)

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        with torch.no_grad():
            if data_parallel: # use Data Parallelism with Multi-GPU
                model = nn.DataParallel(model)

            results = [] # prediction results
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for batch in iter_bar:
                batch = [t.to(self.device) for t in batch]
                with torch.no_grad(): # evaluation without gradient calculation
                    accuracy, result = evaluate(model, batch) # accuracy to print
                results.append(result)

                iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        return results

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                print('Loading the pretrained model from', pretrain_file)
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts


    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

class MLMTrainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name
        self.cross_ent = nn.CrossEntropyLoss(reduction='none')
        self.sent_cross_ent = nn.CrossEntropyLoss()


    def train(self, writer=None, model_file=None, data_parallel=False):
        """ Train Loop """
        if isinstance(self.model, tuple):
            for m in self.model:
                m.train() # train mode
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss, loss_lm, _ = generator_loss(model, batch, global_step, 
                    self.optimizer,
                    self.cross_ent, self.sent_cross_ent,
                    writer, prefix='train'
                    )
                if data_parallel:
                    loss = loss.mean()
                with torch.autograd.detect_anomaly():
                    loss.backward()

                loss_detached = loss.detach()
                del loss

                global_step += 1
                loss_sum = loss_sum + loss_detached
                iter_bar.set_description('Iter (loss=%5.3f)'%loss_detached)

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                accuracy, result = evaluate(model, batch) # accuracy to print
            results.append(result)

            iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        return results

    def load(self, model_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

    def save(self, i):
        """ save current model """
        torch.save(self.model, os.path.join(self.save_dir, 'backbone.pt'))
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))


class AdversarialTrainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, discriminator, generator, data_iter, optimizer, ratio, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.discriminator = discriminator
        self.generator = generator
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.ratio = ratio
        self.save_dir = save_dir
        self.device = device # device name
        self.d_bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.cross_ent = nn.CrossEntropyLoss(reduction='none')
        self.sent_cross_ent = nn.CrossEntropyLoss()

    def train(self, writer=None, model_file=None, data_parallel=False):
        """ Train Loop """
        self.discriminator.train() # train mode
        self.generator.train()
        self.load(model_file)
        generator = self.generator.to(self.device)
        discriminator = self.discriminator.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            discriminator = nn.DataParallel(discriminator)
            generator = nn.DataParallel(generator)

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = torch.torch(0.) # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                g_loss, generate_logits, _ = generator_loss(generator, batch, global_step, 
                    self.optimizer,
                    self.cross_ent, self.sent_cross_ent,
                    writer, prefix='train'
                    )
                # g_loss.backward()
                if data_parallel:
                    g_loss.mean()

                global_step += 1
                # self.d_optimizer.zero_grad()
                batch[3] = torch.argmax(generate_logits, dim=2).detach()
                _, lm_loss, nsp_loss = discriminator_loss(generator, discriminator, batch, global_step, 
                    self.optimizer,
                    self.d_bce_loss, self.sent_cross_ent,
                    writer, prefix='train')
                d_loss = lm_loss*self.ratio + nsp_loss
                if data_parallel:
                    d_loss.mean()

                total_loss = g_loss + d_loss

                loss_sum = loss_sum + total_loss.detach()

                total_loss.backward()

                self.optimizer.step()

                iter_bar.set_description('(d_loss: {:5.3f}, g_loss: {:5.3f}, loss: {:5.3f})'.format( float(d_loss.detach()), float(g_loss.detach()), float(total_loss.detach()) ) )
                del total_loss

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.generator.eval() # evaluation mode
        self.load(model_file)
        generator = self.generator.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            generator = nn.DataParallel(generator)

        results = [] # prediction results
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                accuracy, result = evaluate(generator, batch) # accuracy to print
            results.append(result)

            iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        return results

    def load(self, model_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.generator.load_state_dict(torch.load(model_file))

    def save(self, i):
        """ save current model """
        torch.save(self.generator, os.path.join(self.save_dir, 'g_backbone.pt'))
        torch.save(self.discriminator, os.path.join(self.save_dir, 'd_backbone.pt'))
        torch.save(self.discriminator.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'd_model_steps_'+str(i)+'.pt'))
