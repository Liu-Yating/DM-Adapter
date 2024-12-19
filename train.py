import os
import os.path as op
import torch
import numpy as np
import random
import time


from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_parameter(model):
    trainable = 0.0
    total = 0.0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            print(name)
            trainable += param.numel()
    print("Total Param: {:.2f}M".format(total//1e6), "Trainable Param: {:.2f} M".format(trainable//1e6) )
    print("Total Param: {:.2f}".format(total), "Trainable Param: {:.2f} ".format(trainable) )
    
if __name__ == '__main__':

    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)

    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
        
    # frezze clip
    name_no_update = ["base_model"]
    for name, param in model.named_parameters():
        if any(ntu in name for ntu in name_no_update):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
    name_update = ["adapter_mlp", "ln_3", "experts", "feed_forward", "ln_4", "param", "v2i_proj", "task_param"]
    for name, param in model.named_parameters():
        if any(ntu in name for ntu in name_update):
            param.requires_grad_(True)

    # Double check
    enabled = set()
    disabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
        else:
            disabled.add(name)
    print(f"Parameters to be updated: {enabled}")
    print(f"-----" * 30)
    print(f"Parameters not to be updated: {disabled}")
    
    #### output parameter
    get_parameter(model)
    
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_time = time.time()
    top1 = evaluator.eval(model.eval())
    end_time = time.time()
    logger.info( "test done. Time: {:.3f}[s]".format(end_time-start_time))

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']


    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
