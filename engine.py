                                                  
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import datetime
import logging
import math
import os
import time
import sys
import gc
from tqdm import tqdm
import numpy as np
import pickle as pk

from dtcc_loss import dtcc_pc_img_text

from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.misc import SmoothedValue
from utils.dist import (
 all_gather_dict,
 all_reduce_average,
 all_reduce_sum,
 is_primary,
 reduce_dict,
 barrier,
)
from models.DETR.util.misc import NestedTensor


def _move_to_cpu(obj):
	if torch.is_tensor(obj):
		return obj.detach().cpu()
	if isinstance(obj, dict):
		return {key: _move_to_cpu(value) for key, value in obj.items()}
	if isinstance(obj, list):
		return [_move_to_cpu(value) for value in obj]
	if isinstance(obj, tuple):
		return tuple(_move_to_cpu(value) for value in obj)
	return obj


def compute_learning_rate(args, curr_epoch_normalized):
	assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
	if (
	 curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
	 and args.warm_lr_epochs > 0
	):
                 
		curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
		 (args.base_lr - args.warm_lr) / args.warm_lr_epochs
		)
	else:
                                 
		curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
		 1 + math.cos(math.pi * curr_epoch_normalized)
		)
	return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
	curr_lr = compute_learning_rate(args, curr_epoch)
	for param_group in optimizer.param_groups:
		param_group["lr"] = curr_lr
	return curr_lr

def train_one_epoch_3dov(
 args,
 curr_epoch,
 model,
 optimizer,
 criterion,
 dataset_config,
 dataset_loader,
 logger,):

	curr_iter = curr_epoch * len(dataset_loader)
	max_iters = args.max_epoch * len(dataset_loader)
	net_device = next(model.parameters()).device

	time_delta = SmoothedValue(window_size=10)
	loss_avg = SmoothedValue(window_size=10)

	model.train()
	barrier()

	for batch_idx, batch_data_label in enumerate(dataset_loader):
		curr_time = time.time()
		curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
		for key in batch_data_label:
			if isinstance(batch_data_label[key], dict) or isinstance(batch_data_label[key], list):
				continue
			batch_data_label[key] = batch_data_label[key].to(net_device)

		img_input = {"image": batch_data_label["image"],
		    "calib": batch_data_label["calib"],}
      
		pc_input = {
		 "point_clouds": batch_data_label["point_clouds"],
		 "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
		 "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
		 "Aug_param": batch_data_label["Aug_param"]
		}

		optimizer.zero_grad()
		if args.ngpus > 1:
			training_data, patches_num = model.module.collect_training_data(pc_input, img_input, criterion, batch_data_label, training=True)
		else:
			training_data, patches_num = model.collect_training_data(pc_input, img_input, criterion, batch_data_label, training=True)
		del img_input
		del pc_input
		gc.collect()
		torch.cuda.empty_cache()
		training_data['pc_feats'] = Variable(training_data['pc_feats'], requires_grad=True)
		loss = model(training_data, patches_num, training=True)


		loss.backward()
		optimizer.step()

          
		time_delta.update(time.time() - curr_time)
		loss_reduced = all_reduce_average(loss)
		loss_avg.update(loss_reduced.item())

		del training_data
		del loss
		gc.collect()
		torch.cuda.empty_cache()
  
		if is_primary() and curr_iter % args.log_every == 0:
			mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
			eta_seconds = (max_iters - curr_iter) * time_delta.avg
			eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
			print(
			 f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
			)
                                                                              

			train_dict = {}
			train_dict["lr"] = curr_lr
			train_dict["memory"] = mem_mb
			train_dict["loss"] = loss_avg.avg
			train_dict["batch_time"] = time_delta.avg
   
			logger.log_scalars(train_dict, curr_iter, prefix="Train/")

		curr_iter += 1
		barrier()

	return

def train_one_epoch_global(
 args,
 curr_epoch,
 model,
 optimizer,
 criterion,
 dataset_config,
 dataset_loader,
 logger,):

	curr_iter = curr_epoch * len(dataset_loader)
	max_iters = args.max_epoch * len(dataset_loader)
	net_device = next(model.parameters()).device

	time_delta = SmoothedValue(window_size=10)
	loss_avg = SmoothedValue(window_size=10)

	model.train()
	barrier()

	for batch_idx, batch_data_label in enumerate(dataset_loader):
		curr_time = time.time()
		curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
		for key in batch_data_label:
			if isinstance(batch_data_label[key], dict) or isinstance(batch_data_label[key], list):
				continue
			batch_data_label[key] = batch_data_label[key].to(net_device)

		pc_input = {
		 "point_clouds": batch_data_label["point_clouds"],
		 "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
		 "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
		 "Aug_param": batch_data_label["Aug_param"]
		}

		optimizer.zero_grad()
		if args.ngpus > 1:
			training_data = model.module.collect_training_data_global(pc_input, batch_data_label, training=True)
		else:
			training_data = model.collect_training_data_global(pc_input, batch_data_label, training=True)
		del pc_input
		gc.collect()
		torch.cuda.empty_cache()
		training_data['pc_feats'] = Variable(training_data['pc_feats'], requires_grad=True)
		if args.ngpus > 1:
			loss = model.module.forward_global(training_data, training=True)
		else:
			loss = model.forward_global(training_data, training=True)

		loss.backward()
		optimizer.step()

          
		time_delta.update(time.time() - curr_time)
		loss_reduced = all_reduce_average(loss)
		loss_avg.update(loss_reduced.item())

		del training_data
		del loss
		gc.collect()
		torch.cuda.empty_cache()
  
		if is_primary() and curr_iter % args.log_every == 0:
			mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
			eta_seconds = (max_iters - curr_iter) * time_delta.avg
			eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
			print(
			 f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
			)
                                                                              

			train_dict = {}
			train_dict["lr"] = curr_lr
			train_dict["memory"] = mem_mb
			train_dict["loss"] = loss_avg.avg
			train_dict["batch_time"] = time_delta.avg
   
			logger.log_scalars(train_dict, curr_iter, prefix="Train/")

		curr_iter += 1
		barrier()

	return

def train_one_epoch_bin(
 args,
 curr_epoch,
 model,
 optimizer,
 criterion,
 dataset_config,
 dataset_loader,
 logger,):

	curr_iter = curr_epoch * len(dataset_loader)
	max_iters = args.max_epoch * len(dataset_loader)
	net_device = next(model.parameters()).device

	time_delta = SmoothedValue(window_size=10)
	loss_avg = SmoothedValue(window_size=10)

	model.train()
	barrier()

	for batch_idx, batch_data_label in enumerate(dataset_loader):
		curr_time = time.time()
		curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
		for key in batch_data_label:
			if isinstance(batch_data_label[key], dict) or isinstance(batch_data_label[key], list):
				continue
			batch_data_label[key] = batch_data_label[key].to(net_device)

		pc_input = {
		 "point_clouds": batch_data_label["point_clouds"],
		 "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
		 "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
		 "Aug_param": batch_data_label["Aug_param"]
		}

		optimizer.zero_grad()
		if args.ngpus > 1:
			loss = model.module.train_bin(pc_input, criterion, batch_data_label)
		else:
			loss = model.train_bin(pc_input, criterion, batch_data_label)
		del pc_input
		gc.collect()
		torch.cuda.empty_cache()

		loss.backward()
		optimizer.step()

          
		time_delta.update(time.time() - curr_time)
		loss_reduced = all_reduce_average(loss)
		loss_avg.update(loss_reduced.item())

		del loss
		gc.collect()
		torch.cuda.empty_cache()
  
		if is_primary() and curr_iter % args.log_every == 0:
			mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
			eta_seconds = (max_iters - curr_iter) * time_delta.avg
			eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
			print(
			 f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
			)
                                                                              

			train_dict = {}
			train_dict["lr"] = curr_lr
			train_dict["memory"] = mem_mb
			train_dict["loss"] = loss_avg.avg
			train_dict["batch_time"] = time_delta.avg
   
			logger.log_scalars(train_dict, curr_iter, prefix="Train/")

		curr_iter += 1
		barrier()

	return

@torch.no_grad()
def evaluate_3dov(
 args,
 curr_epoch,
 model,
 criterion,
 dataset_config,
 dataset_loader,
 logger,
 curr_train_iter,
):

                                                                                                     
	ap_calculator = APCalculator(
	 dataset_config=dataset_config,
	 ap_iou_thresh=[0.25],
	 class2type_map=dataset_config.eval_class2type,
	 exact_eval=False,
	)

	curr_iter = 0
	net_device = next(model.parameters()).device
	num_batches = len(dataset_loader)

	time_delta = SmoothedValue(window_size=10)
	loss_avg = SmoothedValue(window_size=10)
	model.eval()
	barrier()
	epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

	for batch_idx, batch_data_label in enumerate(dataset_loader):
		loss_str = ""
		curr_time = time.time()
		for key in batch_data_label:
			if isinstance(batch_data_label[key], dict) or isinstance(batch_data_label[key], list):
				continue
			batch_data_label[key] = batch_data_label[key].to(net_device)
   
		pc_input = {
		 "point_clouds": batch_data_label["point_clouds"],
		 "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
		 "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
		}
  
		if args.ngpus == 1:
			outputs = model.forward_eval(pc_input)
		else:
			outputs = model.module.forward_eval(pc_input)

                
		loss_str = ""
		if criterion is not None:
			loss, loss_dict = criterion(outputs, batch_data_label)

			loss_reduced = all_reduce_average(loss)
			loss_dict_reduced = reduce_dict(loss_dict)
			loss_avg.update(loss_reduced.item())
			loss_str = f"Loss {loss_avg.avg:0.2f};"

                                                                         
		outputs["outputs"] = all_gather_dict(outputs["outputs"])
		batch_data_label = all_gather_dict(batch_data_label)
  
		ap_calculator.step_meter(outputs, batch_data_label)
		time_delta.update(time.time() - curr_time)
		if is_primary() and curr_iter % args.log_every == 0:
			mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
			print(
			 f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
			)

			test_dict = {}
			test_dict["memory"] = mem_mb
			test_dict["batch_time"] = time_delta.avg
			if criterion is not None:
				test_dict["loss"] = loss_avg.avg
		curr_iter += 1
		barrier()
	if is_primary():
		if criterion is not None:
			logger.log_scalars(
			 loss_dict_reduced, curr_train_iter, prefix="Test_details/"
			)
		logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

	np.savetxt("pred_scene_gt_obj_list_scannet_45.txt", ap_calculator.pred_scene_gt_obj, fmt="%d", delimiter="\t")

	return ap_calculator
