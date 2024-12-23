# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from glob import glob

import monai.networks
import monai.transforms
import numpy as np
import monai
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import GliomaDataset, DeepLesionDataset
from model import get_model
from loss import DINOLoss

import utils
from _delete.vision_transformer import DINOHead

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='ResNext50', type=str,
                        choices=['DenseNet121', 'ResNext50', 'ResNet50'], help="""Name of architecture to train. For quick 
                        experiments with ViTs, we recommend using vit_tiny or vit_small.""")
    # parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
    #                     of input square patches - default 16 (for 16x16 patches). Using smaller
    #                     values leads to better performance but requires more memory. Applies only
    #                     for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
    #                     mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=128, type=int, help="""Dimensionality of
                        the DINO head output. For complex and large datasets large values (like 65536) work well.""")
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
                        Not normalizing leads to better performance but can make the training unstable.
                        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
                        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
                        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
                        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
                        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
                        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
                        to use half precision for training. Improves training time and memory requirements,
                        but can provoke instability and slight decay of performance. We recommend disabling
                        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
                        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
                        weight decay. We use a cosine schedule for WD and using a larger decay by
                        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
                        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
                        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=48, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
                        during which we keep the output layer fixed. Typically doing so during
                        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the end of
                        linear warmup (highest LR used during training). The learning rate is linearly scaled
                        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=25, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
                        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'], 
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
                        local views to generate. Set this parameter to 0 to disable multi-crop training.
                        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for small local view cropping of multi-crop.""")

    # Misc
    # parser.add_argument('--data_path', default='/media/johannes/WD Elements/DINO-Glioblastoma-SSL/dino-original-facebook/tiny-imagenet-200/train/', 
    #                     type=str, help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./results/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, 
                        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args, sequence, dataset_name):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True  

    # load dataset

    if dataset_name == "deeplesion":
        dataset = DeepLesionDataset(args=args, path="/home/moonsurfer/Code/foundation-cancer-image-biomarker/data/preprocessing/deeplesion/annotations/pretrain.csv")
    else:
        dataset = GliomaDataset(args=args, sequence=sequence)
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"Data loaded: there are {len(dataset)} images.")    

    # load student and teacher
    student, teacher, embed_dim = get_model(model_name=args.arch, sequence=sequence)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(in_dim=embed_dim, out_dim=args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))
    teacher = utils.MultiCropWrapper(teacher, DINOHead(in_dim=embed_dim, out_dim=args.out_dim, use_bn=args.use_bn_in_head))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        print("We use Automatic Mixed Precision Training!")
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, f"{args.arch}_{dataset_name}_{sequence}_checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    loss_list = []
    lr_list = []
    wd_list = []
    student_var_list = []
    teacher_var_list = []
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats, embeddings_list = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                                       data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                                       epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        
        utils.save_on_master(save_dict, os.path.join(args.output_dir, f'{args.arch}_{dataset_name}_{sequence}_checkpoint.pth'))

        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'{args.arch}_{dataset_name}_{sequence}_checkpoint{epoch:04}.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if utils.is_main_process():
            with (Path(args.output_dir) / f"{args.arch}_{dataset_name}_{sequence}_train_stats.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        loss_list.append(train_stats["loss"])
        lr_list.append(train_stats["lr"])
        wd_list.append(train_stats["wd"])

        variance = torch.var(torch.concatenate(embeddings_list[0], dim=0), dim=0, unbiased=False)
        student_var_list.append(variance.mean().item())

        variance = torch.var(torch.concatenate(embeddings_list[1], dim=0), dim=0, unbiased=False)
        teacher_var_list.append(variance.mean().item())

        plt.figure(figsize=(18, 10))
        plt.subplot(5, 1, 1)
        plt.plot(np.arange(len(loss_list)), loss_list, label="train")
        plt.title("Loss")
        plt.grid()
        plt.legend()

        plt.subplot(5, 1, 2)
        plt.plot(np.arange(len(lr_list)), lr_list, label="train")
        plt.title("Learning Rate")
        plt.grid()
        plt.legend()

        plt.subplot(5, 1, 3)
        plt.plot(np.arange(len(wd_list)), wd_list, label="train")
        plt.title("Weight Decay")
        plt.grid()
        plt.legend()

        plt.subplot(5, 1, 4)
        plt.plot(np.arange(len(student_var_list)), student_var_list, label="train")
        plt.title("Student Embeddding Variance")
        plt.grid()
        plt.legend()

        plt.subplot(5, 1, 5)
        plt.plot(np.arange(len(teacher_var_list)), teacher_var_list, label="train")
        plt.title("Teacher Embedding Variance")
        plt.grid()
        plt.legend()

        plt.savefig(Path(args.output_dir) / f"{args.arch}_{dataset_name}_{sequence}_train_stats.png")
        plt.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    
    student_outputs = []
    teacher_outputs = []
    diff_outputs = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
    # for it, (images1, images2) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        # images = [images1, images2]
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        
        if fp16_scaler is not None:
            with torch.amp.autocast('cuda'):
                teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)
        else:
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        
        student_outputs.append(student_output.detach())
        teacher_outputs.append(teacher_output.detach())

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, [student_outputs, teacher_outputs]





if __name__ == '__main__':

    # time.sleep(21600)

    # T1, T2, T1T1, T2T2, T1T2, T1T2T1T2

    for architecture in ["ResNext50"]:
        for dataset in ["deeplesion"]:
            for sequence in ["T1"]:

                parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
                # args = parser.parse_args()
                args, unknown = parser.parse_known_args()
                args.arch = architecture
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                train_dino(args, sequence, dataset)
