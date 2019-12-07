"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

from random import randint, shuffle
from random import random as rand

import numpy as np
import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter
import os
import multiprocessing as mp
import tokenization
import models
import optim
import train
from utils import set_seeds, get_device
from torch.utils.data import Dataset, DataLoader
from data import seek_random_offset, SentPairDataset, Pipeline, Preprocess4Pretrain, seq_collate


class Generator(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.hidden, 2)

        # decoder is shared with embedding layer
        ## project hidden layer to embedding layer
        embed_weight2 = self.transformer.embed.tok_embed2.weight
        n_hidden, n_embedding = embed_weight2.size()
        self.decoder1 = nn.Linear(n_hidden, n_embedding, bias=False)
        self.decoder1.weight.data = embed_weight2.data.t()

        ## project embedding layer to vocabulary layer
        embed_weight1 = self.transformer.embed.tok_embed1.weight
        n_vocab, n_embedding = embed_weight1.size()
        self.decoder2 = nn.Linear(n_embedding, n_vocab, bias=False)
        self.decoder2.weight = embed_weight1

        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        logits_lm = self.decoder2(self.decoder1(h_masked)) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf


class Discriminator(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.hidden, 2)

        # decoder is shared with embedding layer
        ## project hidden layer to embedding layer
        self.discriminator = nn.Linear(cfg.hidden, 1, bias=False)
        # self.discriminator.weight = embed_weight1

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        cls_h = self.activ1(self.fc(h[:, 0]))

        logits = self.discriminator(h)
        logits_clsf = self.classifier(cls_h)

        return logits, logits_clsf

class ELECTRA():

    def __init__(self, args):
        self.args = args
        cfg = train.Config.from_json(args.train_cfg)
        model_cfg = models.Config.from_json(args.model_cfg)
        generator_cfg = models.Config.from_json(args.generator_cfg)
        assert model_cfg.max_len == generator_cfg.max_len
        set_seeds(cfg.seed)

        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab, do_lower_case=True)
        tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

        pipeline = [Preprocess4Pretrain(args.max_pred,
                                        args.mask_prob,
                                        list(tokenizer.vocab.keys()),
                                        tokenizer.convert_tokens_to_ids,
                                        model_cfg.max_len,
                                        args.mask_alpha,
                                        args.mask_beta,
                                        args.max_gram)]
        data_iter = DataLoader(SentPairDataset(args.data_file,
                                    cfg.batch_size,
                                    tokenize,
                                    model_cfg.max_len,
                                    pipeline=pipeline), 
                                batch_size=cfg.batch_size, 
                                collate_fn=seq_collate,
                                num_workers=mp.cpu_count())

        discriminator = Discriminator(model_cfg)
        generator = Generator(generator_cfg)

        self.optimizer = optim.optim4GPU(cfg, generator, discriminator)
        # self.g_optimizer = optim.optim4GPU(cfg, generator)
        self.trainer = train.AdversarialTrainer(cfg, 
            discriminator, generator, 
            data_iter, 
            self.optimizer, args.ratio, args.save_dir, get_device())
        os.makedirs(os.path.join(args.log_dir, args.name), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.name)) # for tensorboardX



    def train(self):
        self.trainer.train(self.writer, model_file=None, data_parallel=False)

class MaskTrainer():

    def __init__(self, args):
        self.args = args
        cfg = train.Config.from_json(args.train_cfg)
        model_cfg = models.Config.from_json(args.model_cfg)
        set_seeds(cfg.seed)

        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab, do_lower_case=True)
        tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

        pipeline = [Preprocess4Pretrain(args.max_pred,
                                        args.mask_prob,
                                        list(tokenizer.vocab.keys()),
                                        tokenizer.convert_tokens_to_ids,
                                        model_cfg.max_len,
                                        args.mask_alpha,
                                        args.mask_beta,
                                        args.max_gram)]
        data_iter = DataLoader(SentPairDataset(args.data_file,
                                    cfg.batch_size,
                                    tokenize,
                                    model_cfg.max_len,
                                    pipeline=pipeline), 
                                batch_size=cfg.batch_size, 
                                collate_fn=seq_collate,
                                num_workers=mp.cpu_count())

        model = Generator(model_cfg)

        self.optimizer = optim.optim4GPU(cfg, model)
        self.trainer = train.MLMTrainer(cfg, 
            model, 
            data_iter, 
            self.optimizer, args.save_dir, get_device())
        os.makedirs(os.path.join(args.log_dir, args.name), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.name)) # for tensorboardX

    def train(self):
        self.trainer.train(self.writer, model_file=None, data_parallel=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ALBERT Language Model')
    parser.add_argument('--data_file', type=str, default='./data/wiki.train.tokens')
    parser.add_argument('--vocab', type=str, default='./data/vocab.txt')
    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--ratio', type=float, default=50)
    parser.add_argument('--mode', type=str, choices=['mask','electra'], default='mask')
    parser.add_argument('--train_cfg', type=str, default='./config/pretrain.json')
    parser.add_argument('--generator_cfg', type=str, default='./config/albert_unittest.json')
    parser.add_argument('--model_cfg', type=str, default='./config/albert_unittest.json')

    # official google-reacher/bert is use 20, but 20/512(=seq_len)*100 make only 3% Mask
    # So, using 76(=0.15*512) as `max_pred`
    parser.add_argument('--max_pred', type=int, default=76, help='max tokens of prediction')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='masking probability')

    # try to n-gram masking SpanBERT(Joshi et al., 2019)
    parser.add_argument('--mask_alpha', type=int,
                        default=4, help="How many tokens to form a group.")
    parser.add_argument('--mask_beta', type=int,
                        default=1, help="How many tokens to mask within each group.")
    parser.add_argument('--max_gram', type=int,
                        default=3, help="number of max n-gram to masking")

    parser.add_argument('--save_dir', type=str, default='./saved')
    parser.add_argument('--log_dir', type=str, default='./log')

    args = parser.parse_args()
    if args.mode == 'mask':
        print('Mask pretraining')
        trainer = MaskTrainer(args=args)
    else:
        print('Electra pretraining')
        trainer = ELECTRA(args=args)
    trainer.train()

