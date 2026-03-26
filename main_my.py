from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import torch.distributed as dist
import numpy as np
import random
import os
import json
from args import get_args
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import EagleNet
from modules.optimization import BertAdam

from util import get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
from train_and_eval import train_epoch, eval_epoch

num_cuda_visible_devices = torch.cuda.device_count()
if num_cuda_visible_devices > 1:
    from datetime import timedelta
    dist.init_process_group(backend="nccl", 
                            timeout=timedelta(seconds=7200000))  # was 1800000


import time
global logger



def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if num_cuda_visible_devices > 1:
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)
        args.world_size = world_size
        rank = torch.distributed.get_rank()
        args.rank = rank
    # else:
    #     torch.cuda.set_device(int(os.environ['CUDA_VISIBLE_DEVICES']))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if num_cuda_visible_devices == 1 or args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    if num_cuda_visible_devices == 1 or local_rank == 0:
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    if n_gpu > 1:
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    else:
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
    model = EagleNet.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    if n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer=None, tr_loss=None, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    if optimizer is not None:
        optimizer_state_file = os.path.join(
            args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
        torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': tr_loss,
                }, optimizer_state_file)
        logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if num_cuda_visible_devices == 1 or args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        if n_gpu > 1:
            cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        else:
            cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
        model = EagleNet.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model



def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    if args.best_ckpt_path == "":
        model = init_model(args, device, n_gpu, args.local_rank)
    else:
        model = load_model(-1, args, n_gpu, device, model_file=args.best_ckpt_path)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length



    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if num_cuda_visible_devices == 1 or args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
        # R1, _, _ = eval_epoch(args, model, test_dataloader, device, n_gpu, logger)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if num_cuda_visible_devices == 1 or args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']

        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, logger, local_rank=args.local_rank)
            if num_cuda_visible_devices == 1 or args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = None
                # Uncomment if want to save checkpoint
                # output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")
                output_model_file = save_model(epoch, args, model, None, tr_loss, type_name="")

                if args.eval_in_train:
                    R1, _, _ = eval_epoch(args, model, test_dataloader, device, n_gpu, logger)
                else:
                    R1 = 0.
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

        # test on the best checkpoint after training
        if num_cuda_visible_devices == 1 or args.local_rank == 0:
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            R1, sim_matrix, time = eval_epoch(args, model, test_dataloader, device, n_gpu, logger)
            # np.save(os.path.join(args.output_dir, "sim_matrix.npy"), sim_matrix)
            eval_time = {'cache_feature_time': time[0], 'compute_sim_time': time[1], 'total_time': time[0] + time[1]}
            json.dump(eval_time, open(os.path.join(args.output_dir, "eval_time.json"), 'w'), indent=4)

    elif args.do_eval:
        if num_cuda_visible_devices == 1 or args.local_rank == 0:
            R1, sim_matrix, time = eval_epoch(args, model, test_dataloader, device, n_gpu, logger)
            # np.save(os.path.join(args.output_dir, "sim_matrix.npy"), sim_matrix)
            eval_time = {'cache_feature_time': time[0], 'compute_sim_time': time[1], 'total_time': time[0] + time[1]}
            json.dump(eval_time, open(os.path.join(args.output_dir, "eval_time.json"), 'w'), indent=4)

if __name__ == "__main__":
    main()