                                                  

import argparse
import os
import sys
import pickle

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

                                 
from local_datasets import build_dataset
from engine import evaluate_3dov, train_one_epoch_3dov, train_one_epoch_bin, train_one_epoch_global
from models import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible
from utils.logger import Logger

from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

 


def make_args_parser():
	parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

                           

                      
	parser.add_argument("--base_lr", default=5e-4, type=float)
	parser.add_argument("--warm_lr", default=1e-6, type=float)
	parser.add_argument("--warm_lr_epochs", default=9, type=int)
	parser.add_argument("--final_lr", default=1e-5, type=float)
	parser.add_argument("--lr_scheduler", default="cosine", type=str)
	parser.add_argument("--weight_decay", default=0.1, type=float)
	parser.add_argument("--filter_biases_wd", default=False, action="store_true")
	parser.add_argument(
	 "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
	)

                  
	parser.add_argument(
	 "--model_name",
	 default="3detr",
	 type=str,
	 help="Name of the model",
	 choices=["3detr"],
	)
            
	parser.add_argument(
	 "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
	)
                                                   
	parser.add_argument("--enc_nlayers", default=3, type=int)
	parser.add_argument("--enc_dim", default=256, type=int)
	parser.add_argument("--enc_ffn_dim", default=128, type=int)
	parser.add_argument("--enc_dropout", default=0.1, type=float)
	parser.add_argument("--enc_nhead", default=4, type=int)
	parser.add_argument("--enc_pos_embed", default=None, type=str)
	parser.add_argument("--enc_activation", default="relu", type=str)

            
	parser.add_argument("--dec_nlayers", default=8, type=int)
	parser.add_argument("--dec_dim", default=256, type=int)
	parser.add_argument("--dec_ffn_dim", default=256, type=int)
	parser.add_argument("--dec_dropout", default=0.1, type=float)
	parser.add_argument("--dec_nhead", default=4, type=int)

                                            
	parser.add_argument("--mlp_dropout", default=0.3, type=float)
	parser.add_argument(
	 "--nsemcls",
	 default=-1,
	 type=int,
	 help="Number of semantic object classes. Can be inferred from dataset",
	)

                       
	parser.add_argument("--preenc_npoints", default=2048, type=int)
	parser.add_argument(
	 "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
	)
	parser.add_argument("--nqueries", default=256, type=int)
	parser.add_argument("--use_color", default=False, action="store_true")

                     
            
	parser.add_argument("--matcher_giou_cost", default=2, type=float)
	parser.add_argument("--matcher_cls_cost", default=1, type=float)
	parser.add_argument("--matcher_center_cost", default=0, type=float)
	parser.add_argument("--matcher_objectness_cost", default=0, type=float)

                 
	parser.add_argument("--loss_giou_weight", default=0, type=float)
	parser.add_argument("--loss_sem_cls_weight", default=1e-10, type=float)
	parser.add_argument(
	 "--loss_no_object_weight", default=0.2, type=float
	)                                                   
	parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
	parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
	parser.add_argument("--loss_center_weight", default=5.0, type=float)
	parser.add_argument("--loss_size_weight", default=1.0, type=float)

                    
	parser.add_argument(
	 "--dataset_name", required=True, type=str, choices=["scannet"]
	)
	parser.add_argument(
	 "--dataset_root_dir",
	 type=str,
	 default=None,
	 help="Root directory containing the dataset files. \
			  If None, default values from scannet.py are used",
	)
	parser.add_argument(
	 "--meta_data_dir",
	 type=str,
	 default=None,
	 help="Root directory containing the metadata files. \
			  If None, default values from scannet.py are used",
	)
	parser.add_argument("--dataset_num_workers", default=4, type=int)
	parser.add_argument("--batchsize_per_gpu", default=3, type=int)

                     
	parser.add_argument("--start_epoch", default=-1, type=int)
	parser.add_argument("--max_epoch", default=720, type=int)
	parser.add_argument("--eval_every_epoch", default=1000, type=int)
	parser.add_argument("--seed", default=0, type=int)

                    
	parser.add_argument("--test_only", default=False, action="store_true")

                
	parser.add_argument("--checkpoint_dir", default=None, type=str)
	parser.add_argument("--log_every", default=10, type=int)
	parser.add_argument("--log_metrics_every", default=20, type=int)
	parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

                                 
	parser.add_argument("--ngpus", default=1, type=int)
	parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)
 
            
 
                               
	parser.add_argument('--hidden_dim', default=256, type=int,
	     help="Size of the embeddings (dimension of the transformer)")

	parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
	     help="Type of positional embedding to use on top of the image features")

                         
	parser.add_argument('--lr_backbone', default=1e-5, type=float)
 
                  
	parser.add_argument('--masks', action='store_true',
	     help="Train segmentation head if the flag is provided")    
 
	parser.add_argument('--backbone', default='resnet50', type=str,
	     help="Name of the convolutional backbone to use")
 
	parser.add_argument('--dilation', action='store_true',
	     help="If true, we replace stride with dilation in the last convolutional block (DC5)")

	parser.add_argument('--train_bin', action='store_true')
	parser.add_argument('--train_local', action='store_true')
	parser.add_argument('--train_global', action='store_true')        
 
                            
	parser.add_argument('--dropout', default=0.1, type=float,
	     help="Dropout applied in the transformer")
      
	parser.add_argument('--nheads', default=8, type=int,
	     help="Number of attention heads inside the transformer's attentions")
 
	parser.add_argument('--dim_feedforward', default=2048, type=int,
	     help="Intermediate size of the feedforward layers in the transformer blocks")
      
	parser.add_argument('--enc_layers', default=6, type=int,
	     help="Number of encoding layers in the transformer")
      
	parser.add_argument('--dec_layers', default=6, type=int,
	     help="Number of decoding layers in the transformer")
 
	parser.add_argument('--pre_norm', action='store_false')
 
	parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
	     help="Disables auxiliary decoding losses (loss at each layer)")
 
            
	parser.add_argument('--set_cost_class', default=1, type=float,
	     help="Class coefficient in the matching cost")
	parser.add_argument('--set_cost_bbox', default=5, type=float,
	     help="L1 box coefficient in the matching cost")
	parser.add_argument('--set_cost_giou', default=2, type=float,
	     help="giou box coefficient in the matching cost")
 
                                 
	parser.add_argument('--mask_loss_coef', default=1, type=float)
	parser.add_argument('--dice_loss_coef', default=1, type=float)
	parser.add_argument('--bbox_loss_coef', default=5, type=float)
	parser.add_argument('--giou_loss_coef', default=2, type=float)    
	parser.add_argument('--eos_coef', default=0.1, type=float,
	     help="Relative classification weight of the no-object class")
	return parser

def do_train(
 args,
 model,
 model_no_ddp,
 optimizer,
 criterion,
 dataset_config,
 dataloaders,
 best_val_metrics,
):
	num_iters_per_epoch = len(dataloaders["train"])
	num_iters_per_eval_epoch = len(dataloaders["test"])
	print(f"Model is {model}")
	print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
	print(f"One training epoch = {num_iters_per_epoch} iters.")
	print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

	final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
	final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

	if os.path.isfile(final_eval):
		print(f"Found final eval file {final_eval}. Skipping training.")
		return

	logger = Logger(args.checkpoint_dir)

	for epoch in range(args.start_epoch, args.max_epoch):
   
		if is_distributed():
			dataloaders["train_sampler"].set_epoch(epoch)

		if args.train_bin:
			train_one_epoch_bin(
			 args,
			 epoch,
			 model,
			 optimizer,
			 criterion,
			 dataset_config,
			 dataloaders["train"],
			 logger,
			)

			save_checkpoint(
			 args.checkpoint_dir,
			 model_no_ddp.bin_head,
			 optimizer,
			 epoch,
			 args,
			)
		elif args.train_global:
			train_one_epoch_global(
			 args,
			 epoch,
			 model,
			 optimizer,
			 criterion,
			 dataset_config,
			 dataloaders["train"],
			 logger,
			)

			save_checkpoint(
			 args.checkpoint_dir,
			 model_no_ddp.global_projector,
			 optimizer,
			 epoch,
			 args,
			)
		elif args.train_local:
			train_one_epoch_3dov(
			 args,
			 epoch,
			 model,
			 optimizer,
			 criterion,
			 dataset_config,
			 dataloaders["train"],
			 logger,
			)

			save_checkpoint(
			 args.checkpoint_dir,
			 model_no_ddp.projector,
			 optimizer,
			 epoch,
			 args,
			)

def test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders):
	logger = Logger()
	epoch = -1
	curr_iter = 0
	ap_calculator = evaluate_3dov(
	 args,
	 epoch,
	 model,
	 None,
	 dataset_config,
	 dataloaders["test"],
	 logger,
	 curr_iter,
	)
	metrics = ap_calculator.compute_metrics()
	metric_str = ap_calculator.metrics_to_str(metrics)
	if is_primary():
		print("==" * 10)
		print(f"Test model; Metrics {metric_str}")
		print("==" * 10)

def main(local_rank, args):
	if args.ngpus > 1:
		print(
		 "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
		)
		print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
		init_distributed(
		 local_rank,
		 global_rank=local_rank,
		 world_size=args.ngpus,
		 dist_url=args.dist_url,
		 dist_backend="nccl",
		)

	print(f"Called with args: {args}")
	torch.cuda.set_device(local_rank)
	np.random.seed(args.seed + get_rank())
	torch.manual_seed(args.seed + get_rank())
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed + get_rank())

	datasets, dataset_config = build_dataset(args)
 
                              
	model, _ = build_model(args, dataset_config, local_rank)

	train_dtcc_only = True

	for name, param in model.named_parameters():
		if not train_dtcc_only:
			if "img_model" in name:
				param.requires_grad=False
		else:
			if "pc_model.clip_header" not in name:
				param.requires_grad=False

	for name, param in model.named_parameters():
		if name == "img_model.visual.layer4.2.bn1.weight":
			print(param)


	model = model.cuda(local_rank)
	model_no_ddp = model
	if not args.test_only:
		if args.train_bin:
			model.bin_head.train()
			for name, param in model.bin_head.named_parameters():
				param.requires_grad=True
		elif args.train_global:
			model.global_projector.train()
			for name, param in model.global_projector.named_parameters():
				param.requires_grad=True   
		elif args.train_local:
			model.projector.train()
			for name, param in model.projector.named_parameters():
				param.requires_grad=True
	if is_distributed():
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
		model = torch.nn.parallel.DistributedDataParallel(
		 model, device_ids=[local_rank]
		)
	criterion = build_criterion(args, dataset_config)
	criterion = criterion.cuda(local_rank)
 
	dataloaders = {}
	if args.test_only:
		dataset_splits = ["test"]
	else:
		dataset_splits = ["train", "test"]
	for split in dataset_splits:
		if split == "train":
			shuffle = True
		else:
			shuffle = False
		if is_distributed():
			sampler = DistributedSampler(datasets[split], shuffle=shuffle)
		elif shuffle:
			sampler = torch.utils.data.RandomSampler(datasets[split])
		else:
			sampler = torch.utils.data.SequentialSampler(datasets[split])

		dataloaders[split] = DataLoader(
		 datasets[split],
		 sampler=sampler,
		 batch_size=args.batchsize_per_gpu,
		 num_workers=args.dataset_num_workers,
		 worker_init_fn=my_worker_init_fn,
		)
		dataloaders[split + "_sampler"] = sampler

	if args.test_only:
		test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders)
	else:
		assert (
		 args.checkpoint_dir is not None
		), f"Please specify a checkpoint dir using --checkpoint_dir"
		if is_primary() and not os.path.isdir(args.checkpoint_dir):
			os.makedirs(args.checkpoint_dir, exist_ok=True)
		if args.train_bin:
			optimizer = build_optimizer(args, model_no_ddp.bin_head)
		elif args.train_global:
			optimizer = build_optimizer(args, model_no_ddp.global_projector)   
		elif args.train_local:
			optimizer = build_optimizer(args, model_no_ddp.projector)
		else:
			raise ValueError("Please specify a training phase from --train_bin, --train_global, --train_local")
		loaded_epoch, best_val_metrics = resume_if_possible(
		 args.checkpoint_dir, model_no_ddp, optimizer
		)
		args.start_epoch = loaded_epoch + 1

		do_train(
		args,
		model,
		model_no_ddp,
		optimizer,
		criterion,
		dataset_config,
		dataloaders,
		best_val_metrics,
		)


def launch_distributed(args):
	world_size = args.ngpus
	if world_size == 1:
		main(local_rank=0, args=args)
	else:
		torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
	parser = make_args_parser()
	args = parser.parse_args()
                    
                                        
                              
         
	try:
		set_start_method("spawn")
	except RuntimeError:
		pass
	launch_distributed(args)
