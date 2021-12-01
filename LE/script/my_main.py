"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
import random
import my_data_util
from my_data_util import ClothSample
import numpy as np
import torch
import time
from pytorch_pretrained_bert.modeling import BertForCloth
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import functools

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def my_eval():
    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_data=args.eval_data
        # logging("***** Running evaluation *****")
        # logging("  Batch size = {}".format(args.eval_batch_size))
        valid_data = my_data_util.Loader(eval_data, args.cache_size, args.eval_batch_size, device)

        model.eval()
        eval_loss, eval_accuracy, _, _ = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, nb_eval_h_examples = 0, 0, 0
        for inp, tgt in valid_data.data_iter(shuffle=False):

            with torch.no_grad():
                tmp_eval_loss, tmp_eval_accuracy, _, _ = model(inp, tgt)
            if n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean() # mean() to average on multi-gpu.
                tmp_eval_accuracy = tmp_eval_accuracy.sum()
            eval_loss += tmp_eval_loss.item()
            eval_accuracy += tmp_eval_accuracy.item()
            nb_eval_examples += inp[-2].sum().item()
            nb_eval_h_examples += (inp[-2].sum(-1) * inp[-1]).sum().item()
            nb_eval_steps += 1         

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        result = {'valid_eval_loss': eval_loss,
                  'valid_eval_accuracy': eval_accuracy,
                  'global_step': global_step}

        logging('step: {} | valid loss: {} | valid acc {}'.format(
                    global_step, eval_loss, eval_accuracy))

        model.train()

# Define parser
parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--bert_model", 
                    default='bert-base-uncased', 
                    type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

parser.add_argument("--output_dir",
                    default='EXP/',
                    type=str,
                    required=True,
                    help="The output directory where the model checkpoints will be written.")

parser.add_argument("--train_data",
                    default='./data',
                    type=str,
                    help="The training data dir.")

parser.add_argument("--eval_data",
                    default='./data',
                    type=str,
                    help="The evaluation data dir.")

# Other parameters
parser.add_argument("--do_train",
                    default=False,
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--train_batch_size",
                    default=4,
                    type=int,
                    help="Total batch size for training.")

parser.add_argument("--cache_size",
                    default=256,
                    type=int,
                    help="Total batch size for training.")

parser.add_argument("--eval_batch_size",
                    default=1,
                    type=int,
                    help="Total batch size for eval.")

parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")

parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")

parser.add_argument("--num_log_steps",
                    default=10,
                    type=int,
                    help="Total number of training epochs to perform.")

parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")

parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")

parser.add_argument('--seed',
                    type=int,
                    default=24,
                    help="random seed for initialization")

parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument('--loss_scale',
                    type=float, default=128,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

args = parser.parse_args()

# Set log file
suffix = time.strftime('%Y%m%d-%H%M%S')
args.output_dir = os.path.join(args.output_dir, suffix)

if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
os.makedirs(args.output_dir, exist_ok=True)

logging = get_logger(os.path.join(args.output_dir, 'log.txt'))

# Set device
if args.local_rank == -1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

logging("device {} n_gpu {} distributed training {}".format(device, n_gpu, bool(args.local_rank != -1)))

if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                        args.gradient_accumulation_steps))

args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

num_train_steps = None
train_data = None
if args.do_train:
    train_data = my_data_util.Loader(args.train_data, args.cache_size, args.train_batch_size, device)
    num_train_steps = int(
        train_data.data_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

# Prepare model
model = BertForCloth.from_pretrained(args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))

# model=torch.load('./EXP/20210105-165056/model.pt')

model.to(device)

if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
    model, 
    device_ids=[args.local_rank], 
    output_device=args.local_rank)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model, device_ids=[0])

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
t_total = num_train_steps
if args.local_rank != -1:
    t_total = t_total // torch.distributed.get_world_size()
optimizer = BertAdam(optimizer_grouped_parameters,
                        lr=args.learning_rate,
                        warmup=args.warmup_proportion,
                        t_total=t_total)

global_step = 0
if args.do_train:
    logging("***** Running training *****")
    logging("  Batch size = {}".format(args.train_batch_size))
    logging("  Training epoch = {}".format(args.num_train_epochs))
    logging("  Num steps = {}".format(num_train_steps))
    logging("  Model = {}".format(args.bert_model))

    model.train()
    for _ in range(int(args.num_train_epochs)):
        tr_loss = 0
        tr_acc = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for inp, tgt in train_data.data_iter():
            loss, acc, _, _ = model(inp, tgt)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
                acc = acc.sum()
            if args.loss_scale != 1.0:
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            tr_acc += acc.item()
            nb_tr_examples += inp[-2].sum()
            nb_tr_steps += 1
            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1
            if (global_step % args.num_log_steps == 0):
                logging('step: {} | train loss: {} | train acc {}'.format(
                    global_step, tr_loss / nb_tr_examples, tr_acc / nb_tr_examples))
                my_eval()
                tr_loss = 0
                tr_acc = 0
                nb_tr_examples = 0

# save model
save_path=args.output_dir+'/model.pt'
torch.save(model, save_path)

my_eval()