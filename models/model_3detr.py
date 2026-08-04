                                                  
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.functional import cross_entropy
from torch.autograd import Variable
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from utils.pc_util import scale_points, shift_scale_points
from utils.pc_util import write_oriented_bbox
import utils.pc_util as pc_util
import cv2
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoModelForTextEncoding, GenerationConfig
import torchvision.transforms as T
from utils.box_util import extract_pc_in_box3d, box3d_iou
import gc
from random import sample, randint

try:
	from torchvision.transforms import InterpolationMode
	BICUBIC = InterpolationMode.BICUBIC
except ImportError:
	BICUBIC = Image.BICUBIC

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
        TransformerDecoderLayer, TransformerEncoder,
        TransformerEncoderLayer)

from .DETR.backbone import build_backbone
from .DETR.transformer import build_transformer
from .DETR.detr import DETR, SetCriterion, Projector, Global_Projector
from .DETR.matcher import build_matcher

            
import sys
import clip

from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls

from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from config import VICUNA_PATH, CLIP_PATH, LOCALIZATION_PATH, LOCAL_PROJECTOR_PATH, BIN_HEAD_PATH, GLOBAL_PROJECTOR_PATH

class BoxProcessor(object):
\
\
    

	def __init__(self, dataset_config):
		self.dataset_config = dataset_config

	def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
		center_unnormalized = query_xyz + center_offset
		center_normalized = shift_scale_points(
		 center_unnormalized, src_range=point_cloud_dims
		)
		return center_normalized, center_unnormalized

	def compute_predicted_size(self, size_normalized, point_cloud_dims):
		scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
		scene_scale = torch.clamp(scene_scale, min=1e-1)
		size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
		return size_unnormalized

	def compute_predicted_angle(self, angle_logits, angle_residual):
		if angle_logits.shape[-1] == 1:
			angle = angle_logits * 0 + angle_residual * 0
			angle = angle.squeeze(-1).clamp(min=0)
		else:
			angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
			pred_angle_class = angle_logits.argmax(dim=-1).detach()
			angle_center = angle_per_cls * pred_angle_class
			angle = angle_center + angle_residual.gather(
			 2, pred_angle_class.unsqueeze(-1)
			).squeeze(-1)
			mask = angle > np.pi
			angle[mask] = angle[mask] - 2 * np.pi
		return angle

	def compute_objectness_and_cls_prob(self, cls_logits):
		cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
		objectness_prob = 1 - cls_prob[..., -1]
		return cls_prob[..., :-1], objectness_prob

	def box_parametrization_to_corners(
	 self, box_center_unnorm, box_size_unnorm, box_angle
	):
		return self.dataset_config.box_parametrization_to_corners(
		 box_center_unnorm, box_size_unnorm, box_angle
		)

class Model3DETR(nn.Module):
	def __init__(
	 self,
	 pre_encoder,
	 encoder,
	 decoder,
	 dataset_config,
	 encoder_dim=256,
	 decoder_dim=256,
	 position_embedding="fourier",
	 mlp_dropout=0.3,
	 num_queries=256,
	 args=None
	):
		super().__init__()
		self.args = args
		self.pre_encoder = pre_encoder
		self.encoder = encoder
		if hasattr(self.encoder, "masking_radius"):
			hidden_dims = [encoder_dim]
		else:
			hidden_dims = [encoder_dim, encoder_dim]
		self.encoder_to_decoder_projection = GenericMLP(
		 input_dim=encoder_dim,
		 hidden_dims=hidden_dims,
		 output_dim=decoder_dim,
		 norm_fn_name="bn1d",
		 activation="relu",
		 use_conv=True,
		 output_use_activation=True,
		 output_use_norm=True,
		 output_use_bias=False,
		)
		self.pos_embedding = PositionEmbeddingCoordsSine(
		 d_pos=decoder_dim, pos_type=position_embedding, normalize=True
		)
		self.query_projection = GenericMLP(
		 input_dim=decoder_dim,
		 hidden_dims=[decoder_dim],
		 output_dim=decoder_dim,
		 use_conv=True,
		 output_use_activation=True,
		 hidden_use_bias=True,
		)
		self.decoder = decoder
		self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

		self.num_queries = num_queries
		self.box_processor = BoxProcessor(dataset_config)  

	def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
		mlp_func = partial(
		 GenericMLP,
		 norm_fn_name="bn1d",
		 activation="relu",
		 use_conv=True,
		 hidden_dims=[decoder_dim, decoder_dim],
		 dropout=mlp_dropout,
		 input_dim=decoder_dim,
		)

		semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

		center_head = mlp_func(output_dim=3)
		size_head = mlp_func(output_dim=3)
		angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
		angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

		mlp_heads = [
		 ("sem_cls_head", semcls_head),
		 ("center_head", center_head),
		 ("size_head", size_head),
		 ("angle_cls_head", angle_cls_head),
		 ("angle_residual_head", angle_reg_head),
		]
		self.mlp_heads = nn.ModuleDict(mlp_heads)

	def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
		query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
		query_inds = query_inds.long()
		query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
		query_xyz = torch.stack(query_xyz)
		query_xyz = query_xyz.permute(1, 2, 0)

		pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
		query_embed = self.query_projection(pos_embed)
		return query_xyz, query_embed

	def _break_up_pc(self, pc):

		xyz = pc[..., 0:3].contiguous()
		features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
		return xyz, features

	def run_encoder(self, point_clouds):
		xyz, features = self._break_up_pc(point_clouds)
		pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)

		pre_enc_features = pre_enc_features.permute(2, 0, 1)


		enc_xyz, enc_features, enc_inds = self.encoder(
		 pre_enc_features, xyz=pre_enc_xyz
		)
		if enc_inds is None:
			enc_inds = pre_enc_inds
		else:
			enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.long())
		return enc_xyz, enc_features, enc_inds

	def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
\
\
\
\
\
\
\
     
		box_features = box_features.permute(0, 2, 3, 1)
		num_layers, batch, channel, num_queries = (
		 box_features.shape[0],
		 box_features.shape[1],
		 box_features.shape[2],
		 box_features.shape[3],
		)
		box_features = box_features.reshape(num_layers * batch, channel, num_queries)

		cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
		center_offset = (
		 self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
		)
		size_normalized = (
		 self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
		)
		angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
		angle_residual_normalized = self.mlp_heads["angle_residual_head"](
		 box_features
		).transpose(1, 2)

		cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
		center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
		size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
		angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
		angle_residual_normalized = angle_residual_normalized.reshape(
		 num_layers, batch, num_queries, -1
		)
		angle_residual = angle_residual_normalized * (
		 np.pi / angle_residual_normalized.shape[-1]
		)

		outputs = []
		for l in range(num_layers):
			(
			 center_normalized,
			 center_unnormalized,
			) = self.box_processor.compute_predicted_center(
			 center_offset[l], query_xyz, point_cloud_dims
			)
			angle_continuous = self.box_processor.compute_predicted_angle(
			 angle_logits[l], angle_residual[l]
			)
			size_unnormalized = self.box_processor.compute_predicted_size(
			 size_normalized[l], point_cloud_dims
			)
			box_corners = self.box_processor.box_parametrization_to_corners(
			 center_unnormalized, size_unnormalized, angle_continuous
			)

			(
			 semcls_prob,
			 objectness_prob,
			) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

			box_prediction = {
			 "sem_cls_logits": cls_logits[l],
			 "center_normalized": center_normalized.contiguous(),
			 "center_unnormalized": center_unnormalized,
			 "size_normalized": size_normalized[l],
			 "size_unnormalized": size_unnormalized,
			 "angle_logits": angle_logits[l],
			 "angle_residual": angle_residual[l],
			 "angle_residual_normalized": angle_residual_normalized[l],
			 "angle_continuous": angle_continuous,
			 "objectness_prob": objectness_prob,
			 "sem_cls_prob": semcls_prob,
			 "box_corners": box_corners,
			}
			outputs.append(box_prediction)

		aux_outputs = outputs[:-1]
		outputs = outputs[-1]

		return {
		 "outputs": outputs,
		 "aux_outputs": aux_outputs,
		}

	def forward(self, inputs, encoder_only=False):
		point_clouds = inputs["point_clouds"]

		enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
		enc_features = self.encoder_to_decoder_projection(
		 enc_features.permute(1, 2, 0)
		).permute(2, 0, 1)


		if encoder_only:
			return enc_xyz, enc_features.transpose(0, 1)
		point_cloud_dims = [
		 inputs["point_cloud_dims_min"],
		 inputs["point_cloud_dims_max"],
		]
		query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
		enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

		enc_pos = enc_pos.permute(2, 0, 1)
		query_embed = query_embed.permute(2, 0, 1)
		tgt = torch.zeros_like(query_embed)
		box_features = self.decoder(
		 tgt, enc_features, query_pos=query_embed, pos=enc_pos
		)[0]

		pc_query_feat = box_features[-1,:,:,:]

		box_predictions = self.get_box_predictions(
		 query_xyz, point_cloud_dims, box_features
		)
		pc_query_feat = pc_query_feat / pc_query_feat.norm(dim=2, keepdim=True)

		box_predictions["pc_query_feat"] = pc_query_feat
		box_predictions["enc_features"] = torch.mean(pc_query_feat.transpose(0, 1), dim=1)
		return box_predictions

def build_preencoder(args):
	mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
	preencoder = PointnetSAModuleVotes(
	 radius=0.2,
	 nsample=64,
	 npoint=args.preenc_npoints,
	 mlp=mlp_dims,
	 normalize_xyz=True,
	)
	return preencoder


def build_encoder(args):
	if args.enc_type == "vanilla":
		encoder_layer = TransformerEncoderLayer(
		 d_model=args.enc_dim,
		 nhead=args.enc_nhead,
		 dim_feedforward=args.enc_ffn_dim,
		 dropout=args.enc_dropout,
		 activation=args.enc_activation,
		)
		encoder = TransformerEncoder(
		 encoder_layer=encoder_layer, num_layers=args.enc_nlayers
		)
	elif args.enc_type in ["masked"]:
		encoder_layer = TransformerEncoderLayer(
		 d_model=args.enc_dim,
		 nhead=args.enc_nhead,
		 dim_feedforward=args.enc_ffn_dim,
		 dropout=args.enc_dropout,
		 activation=args.enc_activation,
		)
		interim_downsampling = PointnetSAModuleVotes(
		 radius=0.4,
		 nsample=32,
		 npoint=args.preenc_npoints // 2,
		 mlp=[args.enc_dim, 256, 256, args.enc_dim],
		 normalize_xyz=True,
		)
  
		masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
		encoder = MaskedTransformerEncoder(
		 encoder_layer=encoder_layer,
		 num_layers=3,
		 interim_downsampling=interim_downsampling,
		 masking_radius=masking_radius,
		)
	else:
		raise ValueError(f"Unknown encoder type {args.enc_type}")
	return encoder


def build_decoder(args):
	decoder_layer = TransformerDecoderLayer(
	 d_model=args.dec_dim,
	 nhead=args.dec_nhead,
	 dim_feedforward=args.dec_ffn_dim,
	 dropout=args.dec_dropout,
	)
	decoder = TransformerDecoder(
	 decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
	)
	return decoder


class DETR_3D_2D(nn.Module):
	def __init__(self, args, pc_model, projector, bin_head, global_projector, local_rank, dataset_config):
		super().__init__()
		self.class2type = dataset_config.class2type
		self.pc_model = pc_model
		self.projector = projector
		self.bin_head = bin_head
		self.global_projector = global_projector
		self.max_new_tokens = 10
		self.nqueries = args.nqueries

		if args.train_global:
			self.scenes = [ 
			 'bedroom', 
			 'living room',
			 'bathroom',
			 'kitchen',
			 'conference room',
			 'warehouse',
			 'office',
			 'classroom',
			 'library',
			 'dining room',
			 ''
			]

			vicuna_path = VICUNA_PATH
			self.vicuna = AutoModelForCausalLM.from_pretrained(vicuna_path, device_map=local_rank)
			self.vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
			self.vicuna_generation_config = GenerationConfig.from_pretrained(vicuna_path, "generation_config.json")
			vicuna_question = ["Hint:", ". USER: What scene is it? ASSISTANT: A"]
			with torch.no_grad():
				input_ids = [self.vicuna_tokenizer(vicuna_question[0], return_tensors='pt').input_ids.cuda(), self.vicuna_tokenizer(vicuna_question[1], return_tensors='pt').input_ids.cuda()]
				self.pre_question_embeddings = self.vicuna.base_model.embed_tokens(input_ids[0])
				self.post_question_embeddings = self.vicuna.base_model.embed_tokens(input_ids[1])


		if args.train_local:
			self.clip_model, self.clip_processor = clip.load(CLIP_PATH)
			vicuna_path = VICUNA_PATH
			self.vicuna = AutoModelForCausalLM.from_pretrained(vicuna_path, device_map=local_rank)
			self.vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
			self.vicuna_generation_config = GenerationConfig.from_pretrained(vicuna_path, "generation_config.json")
			vicuna_question = ["Hint:", ". USER: What is this? ASSISTANT: A"]
			with torch.no_grad():
				input_ids = [self.vicuna_tokenizer(vicuna_question[0], return_tensors='pt').input_ids.cuda(), self.vicuna_tokenizer(vicuna_question[1], return_tensors='pt').input_ids.cuda()]
				self.pre_question_embeddings = self.vicuna.base_model.embed_tokens(input_ids[0])
				self.post_question_embeddings = self.vicuna.base_model.embed_tokens(input_ids[1])

			self.colors = ['blue ','red ', 'yellow ', 'black ', 'white ', 'green ', 'grey ', 'brown ', 'golden ', 'bright ', 'purple ', 'orange ', 'pink ', 'cyan ', 'silver ', 'little ', 'small ', 'big ', 'large ', 'tall ', 'short ', 'storage ', 'sectional ', 'bathroom ','tool ', 'conference ', 'shower ', 'granite ','wood ','wooden ', 'shower ', 'kitchen ', 'new ', 'old ', 'computer ', 'bunk ', 'plastic ', 'coffee ', 'file ']

		if args.test_only:
			self.clip_model, self.clip_processor = clip.load(CLIP_PATH)
			vicuna_path = VICUNA_PATH
			self.vicuna = AutoModelForCausalLM.from_pretrained(vicuna_path, device_map=local_rank)
			self.vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_path, use_fast=False)
			self.vicuna_generation_config = GenerationConfig.from_pretrained(vicuna_path, "generation_config.json")
			vicuna_question = ["Hint:", ". USER: What is this? ASSISTANT: A"]
			with torch.no_grad():
				input_ids = [self.vicuna_tokenizer(vicuna_question[0], return_tensors='pt').input_ids.cuda(), self.vicuna_tokenizer(vicuna_question[1], return_tensors='pt').input_ids.cuda()]
				self.pre_question_embeddings = self.vicuna.base_model.embed_tokens(input_ids[0])
				self.post_question_embeddings = self.vicuna.base_model.embed_tokens(input_ids[1])
			self.eval_text = ["toilet",
			   "bed",
			   "chair",
			   "sofa",
			   "dresser",
			   "table",
			   "cabinet",
			   "bookshelf",
			   "pillow",
			   "sink",
			   "bathtub",
			   "refrigerator",
			   "desk",
			   "nightstand",
			   "counter",
			   "door",
			   "curtain",
			   "box",
			   "lamp",
			   "bag"]
                                                                                                                                                                                                                                     
   
			for key in self.class2type:
				if self.class2type[key] not in self.eval_text:
					self.eval_text.append(self.class2type[key])

			eval_text_tokens = clip.tokenize(self.eval_text).cuda()
			self.eval_text_feats = self.clip_model.encode_text(eval_text_tokens)
			self.eval_text_feats = self.eval_text_feats / self.eval_text_feats.norm(dim=1, keepdim=True)
			self.eval_text_num = 20
			self.eval_text_label = torch.arange(self.eval_text_num, dtype=torch.int).cuda()

			vicuna_question = ["Hint:", ". USER: What scene is it? ASSISTANT: A"]
			with torch.no_grad():
				input_ids = [self.vicuna_tokenizer(vicuna_question[0], return_tensors='pt').input_ids.cuda(), self.vicuna_tokenizer(vicuna_question[1], return_tensors='pt').input_ids.cuda()]
				self.pre_question_embeddings_scene = self.vicuna.base_model.embed_tokens(input_ids[0])
				self.post_question_embeddings_scene = self.vicuna.base_model.embed_tokens(input_ids[1])
   

	def batch_encode_text(self, text):
		batch_size = 20

		text_num = text.shape[0]
		cur_start = 0
		cur_end = 0

		all_text_feats = []
		while cur_end < text_num:
			cur_start = cur_end
			cur_end += batch_size
			if cur_end >= text_num:
				cur_end = text_num
   
			cur_text = text[cur_start:cur_end,:]
			cur_text_feats = self.img_model.encode_text(cur_text).detach()
			all_text_feats.append(cur_text_feats)

		all_text_feats = torch.cat(all_text_feats, dim=0)
		return all_text_feats

	def accuracy(self, image_features, text_features, img_label):
		image_features = image_features / image_features.norm(dim=1, keepdim=True)
		text_features = text_features / text_features.norm(dim=1, keepdim=True)

		logit_scale_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		logit_scale = logit_scale_.exp()

		logits_per_image = logit_scale * image_features @ text_features.t()
		logits_per_text = logits_per_image.t()

		probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
		rst = np.argmax(probs, axis=1)

		img_label = img_label.reshape([-1]).detach().cpu().numpy()

		diff = rst-img_label
		return


	def classify(self, image_features, text_features, verbose=False, img_label=None):

		logit_scale_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		logit_scale = logit_scale_.exp()

		logits_per_image = logit_scale * image_features @ text_features.t()

		probs_ = logits_per_image.softmax(dim=-1)
		probs, rst = torch.max(probs_, dim=1)

		probs = torch.where(rst<20, probs, torch.zeros_like(probs).cuda())

		if verbose:
			return probs, rst, probs_, logits_per_image
		else:
			return probs, rst

	def my_compute_box_3d(self, center, size, heading_angle):
		R = pc_util.rotz(-1 * heading_angle)
		l, w, h = size
		x_corners = [-l, l, l, -l, -l, l, l, -l]
		y_corners = [w, w, -w, -w, w, w, -w, -w]
		z_corners = [h, h, h, h, -h, -h, -h, -h]
		corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
		corners_3d[0, :] += center[0]
		corners_3d[1, :] += center[1]
		corners_3d[2, :] += center[2]
		return np.transpose(corners_3d)

	def compute_all_bbox_corners(self, pred_bboxes):
		batch_size, bbox_num, _ = pred_bboxes.shape

		all_corners = []
		for cur_bs in range(batch_size):
			cur_bs_corners = []
			for cur_bbox in range(bbox_num):
				cur_center = pred_bboxes[cur_bs, cur_bbox, :3]
				cur_size = pred_bboxes[cur_bs, cur_bbox, 3:6] / 2
				cur_heading = pred_bboxes[cur_bs, cur_bbox, 6]
				cur_corners = self.my_compute_box_3d(cur_center, cur_size, cur_heading)
				cur_bs_corners.append(cur_corners)

			cur_bs_corners = np.stack(cur_bs_corners, axis=0)
			all_corners.append(cur_bs_corners)

		all_corners = np.stack(all_corners, axis=0)

		return all_corners

	def restore_aug(self, pred_bboxes, Aug_param):
\
\
\
\
\
\
     

		pred_bboxes_restore=pred_bboxes.copy()

		batch_size = pred_bboxes.shape[0]
  
		for bs_ind in range(batch_size):
                 
			Aug_Scale = Aug_param["Aug_Scale"][bs_ind, 0, ...].detach().cpu().numpy()
			Aug_Rot_Ang = Aug_param["Aug_Rot_Ang"][bs_ind].detach().cpu().numpy()
			Aug_Rot_Mat = Aug_param["Aug_Rot_Mat"][bs_ind,...].detach().cpu().numpy()
			Aug_Flip = Aug_param["Aug_Flip"][bs_ind].detach().cpu().numpy()
   
			'''
			print(Aug_Scale)
			print(Aug_Rot_Ang)
			print(Aug_Rot_Mat)
			print(Aug_Flip)
			'''

			pred_bboxes_restore[bs_ind, :, :3] /= Aug_Scale
			pred_bboxes_restore[bs_ind, :, 3:6] /= Aug_Scale
			pred_bboxes_restore[bs_ind, :, 6] += Aug_Rot_Ang
			pred_bboxes_restore[bs_ind, :, 0:3] = np.dot(pred_bboxes_restore[bs_ind, :, 0:3], Aug_Rot_Mat)

			if Aug_Flip == 1:
				pred_bboxes_restore[bs_ind, :, 6] = np.pi - pred_bboxes_restore[bs_ind, :, 6]
				pred_bboxes_restore[bs_ind, :, 0] = -1 * pred_bboxes_restore[bs_ind, :, 0]

		return pred_bboxes_restore

	def proj_pointcloud_into_image(self, xyz, pose, calib_K):
		pose = pose.detach().cpu().numpy()
		calib_K = calib_K.detach().cpu().numpy()

		padding = np.ones([xyz.shape[0], 1])
		xyzp = np.concatenate([xyz, padding], axis=1)
		xyz = (pose @ xyzp.T)[:3,:].T

		intrinsic_image = calib_K[:3,:3]
		xyz_uniform = xyz/xyz[:,2:3]
		xyz_uniform = xyz_uniform.T

		uv = intrinsic_image @ xyz_uniform

		uv /= uv[2:3, :]
		uv = np.around(uv).astype(np.int)

		uv = uv.T

		return uv[:,:2]


	def compute_all_bbox_corners_2d(self, pred_bboxes_corners, calib_param):
\
\
\
\
\
\
     

		calib_Rtilt = calib_param["calib_Rtilt"]
		calib_K = calib_param["calib_K"]

		batch_size, box_num, _, _ = pred_bboxes_corners.shape

		all_corners_2d = []
		for bs_ind in range(batch_size):
			cur_calib_Rtilt = calib_Rtilt[bs_ind,:,:]
			cur_calib_K = calib_K[bs_ind, :, :]

			cur_batch_corners_2d = []
			for box_ind in range(box_num):
				cur_corners_3d = pred_bboxes_corners[bs_ind, box_ind, :, :]

				cur_corners_2d = self.proj_pointcloud_into_image(cur_corners_3d, cur_calib_Rtilt, cur_calib_K)
				cur_batch_corners_2d.append(cur_corners_2d)

			cur_batch_corners_2d = np.stack(cur_batch_corners_2d, axis=0)
			all_corners_2d.append(cur_batch_corners_2d)

		all_corners_2d = np.stack(all_corners_2d, axis=0)
		return all_corners_2d

	def img_preprocess(self, img):
		def resize(img, size):
			img_h, img_w, _ = img.shape
			if img_h > img_w:
				new_w = int(img_w * size / img_h)
				new_h = size
			else:
				new_w = size
				new_h = int(img_h * size / img_w)

			dsize = (new_h, new_w)
			transform = T.Resize(dsize, interpolation=BICUBIC)

			img = img.permute([2,0,1])
			img = transform(img)
			img = img.permute([1,2,0])
   
			return img

		def center_crop(img, dim):
\
\
\
\
      
			width, height = img.shape[1], img.shape[0]

			crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
			crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
			mid_x, mid_y = int(width/2), int(height/2)
			cw2, ch2 = int(crop_width/2), int(crop_height/2) 
			crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
			return crop_img

		def padding(img, dim):
			rst_img = torch.zeros([dim[0], dim[1], 3], dtype=img.dtype, device=img.device)
			h, w, _ = img.shape

			top_left = (np.array(dim)/2 - np.array([h,w])/2).astype(np.int)
			down_right = (top_left + np.array([h,w])).astype(np.int)

			rst_img[top_left[0]:down_right[0], top_left[1]:down_right[1], :] = img.clone()
			return rst_img

		img = resize(img, 224)
		img = padding(img, [224,224])
		return img

	def img_denorm(self, img):
		img = img.detach().cpu().numpy()
		imagenet_std = np.array([0.26862954, 0.26130258, 0.27577711])
		imagenet_mean = np.array([0.48145466, 0.4578275, 0.40821073])
		img = (img * imagenet_std + imagenet_mean) * 255

		return img

	def crop_patches(self, image, corners_2d, image_size, corners_3d, point_cloud, pred_bboxes=None):

		def is_valid_patch(top_left, down_right, cur_corner_3d, cur_point_cloud):
			patch_ori_size = down_right - top_left
			patch_ori_size = patch_ori_size[0] * patch_ori_size[1]

			if np.isnan(top_left[0]) or np.isnan(top_left[1]) or np.isnan(down_right[0]) or np.isnan(down_right[1]):
				return False, np.array([0,0]), np.array([10,10])

			if top_left[0] < 0:
				top_left[0] = 0

			if top_left[1] < 0:
				top_left[1] = 0

			if top_left[0] >= img_width:
				return False, np.array([0,0]), np.array([10,10])

			if top_left[1] >= img_height:
				return False, np.array([0,0]), np.array([10,10])

			if down_right[0] > img_width:
				down_right[0] = img_width - 1

			if down_right[1] > img_height:
				down_right[1] = img_height - 1

			if down_right[0] <= 0:
				return False, np.array([0,0]), np.array([10,10])

			if down_right[1] <= 0:
				return False, np.array([0,0]), np.array([10,10])

			patch_fixed_size = down_right - top_left

			if patch_fixed_size[0] < 10 or patch_fixed_size[1] < 10:
				return False, np.array([0,0]), np.array([10,10])

			patch_fixed_size = patch_fixed_size[0] * patch_fixed_size[1]

			if patch_fixed_size/patch_ori_size < 0.8:
				return False, top_left, down_right

			cur_corner_3d = corners_3d[bs_ind, box_ind, :, :]
			pc, _ = extract_pc_in_box3d(cur_point_cloud, cur_corner_3d)
			if pc.shape[0] < 100:
				return False, top_left, down_right
			return True, top_left, down_right


		batch_size, box_num, _, _ = corners_2d.shape

		all_patch = []

		all_valid = np.zeros([batch_size, box_num], dtype=np.bool)

		for bs_ind in range(batch_size):
			cur_bs_img = image[bs_ind, ...]
			img_width = image_size[bs_ind, 0]
			img_height = image_size[bs_ind, 1]

			cur_bs_patch = []

			for box_ind in range(box_num):
				cur_corners_2d = corners_2d[bs_ind, box_ind, :, :]
				top_left = np.min(cur_corners_2d, axis=0)
				down_right = np.max(cur_corners_2d, axis=0)

				valid_flag, top_left, down_right = is_valid_patch(top_left, down_right, corners_3d[bs_ind, box_ind, :, :], point_cloud[bs_ind,:,:])
				all_valid[bs_ind, box_ind] = valid_flag

				top_left = (int(top_left[0]), int(top_left[1]))
				down_right = (int(down_right[0]), int(down_right[1]))

				cur_patch = cur_bs_img[top_left[1]:down_right[1], top_left[0]:down_right[0], :].clone()



				cur_patch = self.img_preprocess(cur_patch)


				cur_bs_patch.append(cur_patch)

    
				'''
				cur_patch = cur_bs_img[top_left[1]:down_right[1], top_left[0]:down_right[0], :]
				cur_patch_ = img_denorm(cur_patch)
				cv2.imwrite("patch_%03d_%03d.jpg"%(bs_ind, box_ind), cur_patch_)


				pair_3d_bbox = np.expand_dims(pred_bboxes[bs_ind, box_ind, ...], axis=0)
				print(pair_3d_bbox.shape)

				pair_3d_bbox[..., 6] *= -1
				write_oriented_bbox(pair_3d_bbox, "pair_3d_bbox_%03d_%03d.ply"%(bs_ind, box_ind))

				print("valid")
				input()
				'''
			cur_bs_patch = torch.stack(cur_bs_patch, dim=0)
			all_patch.append(cur_bs_patch)

		all_patch = torch.stack(all_patch, dim=0)

		all_patch = all_patch.permute(0,1,4,2,3)
		return all_patch, all_valid
 
	def get_blip_answer(self, image, patches_num):
		inputs = self.blip_processor(image, [self.blip_question] * patches_num, return_tensors="pt").to("cuda")
		out = self.blip_model.generate(**inputs, max_new_tokens=self.max_new_tokens)
		answer = self.blip_processor.batch_decode(out, skip_special_tokens=True)
		return answer
  

	def get_clip_answer(self, blip_answers, patches, patches_num):
		picks = []
		for i in range(patches_num):
			if '?' in blip_answers[i] or 'feature' in blip_answers[i] or 'cat' in blip_answers[i] or 'dog' in blip_answers[i] or 'bird' in blip_answers[i] or 'bear' in blip_answers[i] or 'man' in blip_answers[i] or 'ghost' in blip_answers[i] or 'who' in blip_answers[i] or 'where' in blip_answers[i] or 'what' in blip_answers[i] or 'you' in blip_answers[i] or 'person' in blip_answers[i] or 'object' in blip_answers[i]:
				continue
			image_embedding = self.clip_processor(patches[i]).unsqueeze(0).cuda()                
			text_embedding = clip.tokenize(['This is a ' + blip_answers[i], 'This is not a ' + blip_answers[i]], truncate=True).cuda()
			logits_per_image, logits_per_text = self.clip_model(image_embedding, text_embedding)
			if logits_per_image[0][0] > logits_per_image[0][1]:
				picks.append(i)
		return picks

	def collect_valid_pc_img_feat(self, in_pc_feat, valid_3d_bbox, pair_img_cnt, pair_img_porb, pair_img_label, pair_img_feat):
\
\
\
\
\
\
     

		valid_3d_bbox_num = 0
		for bs_ind in range(len(valid_3d_bbox)):
			valid_3d_bbox_num+=len(valid_3d_bbox[bs_ind])
  
		assert valid_3d_bbox_num == pair_img_cnt

		bbox_counter=0

		out_pc_feat = []
		out_label = []
		out_prob = []
		out_img_feat = []
  
  
		for bs_ind in range(len(valid_3d_bbox)):
			cur_bs_valid_bbox = valid_3d_bbox[bs_ind]

			for bbox_ind in range(len(cur_bs_valid_bbox)):
    
				cur_pc_feat = in_pc_feat[cur_bs_valid_bbox[bbox_ind], bs_ind, :]
				cur_label = pair_img_label[bbox_counter]
				cur_prob = pair_img_porb[bbox_counter]
				cur_img_feat = pair_img_feat[bbox_counter, :]

				out_img_feat.append(cur_img_feat)
				out_pc_feat.append(cur_pc_feat)
				out_label.append(cur_label)
				out_prob.append(cur_prob)
				bbox_counter += 1


		out_pc_feat = torch.stack(out_pc_feat, dim=0)
		out_label = torch.stack(out_label, dim=0)
		out_prob = torch.stack(out_prob, dim=0)
		out_img_feat = torch.stack(out_img_feat, dim=0)

		assert valid_3d_bbox_num == pair_img_cnt and valid_3d_bbox_num == bbox_counter
  

		pair_img_output={"pair_img_feat": out_img_feat,
		     "pair_img_label": out_label,
		     "pair_img_prob": out_prob,
		    }
  
		pc_output={"pc_feat": out_pc_feat,
		     "pc_label": out_label,
		     "pc_prob": out_prob,
		    }
		return pair_img_output, pc_output

	def classify_pc(self, pc_query_feat, text_feat, text_num, box_nums):

		box_num, feat_dim = pc_query_feat.shape

		vicuna_pc_feats = self.projector(pc_query_feat)  
		vicuna_pc_feats = vicuna_pc_feats.unsqueeze(1) 
		vicuna_input_base = torch.cat([self.pre_question_embeddings.expand(box_num, -1, -1), vicuna_pc_feats.half(), self.post_question_embeddings.expand(box_num, -1, -1)], dim=1)
		del pc_query_feat
		del vicuna_pc_feats
		gc.collect()
		torch.cuda.empty_cache()
		vicuna_output_tokens = self.vicuna.generate(inputs_embeds=vicuna_input_base, generation_config=self.vicuna_generation_config, max_new_tokens=self.max_new_tokens)
		vicuna_output = self.vicuna_tokenizer.batch_decode(vicuna_output_tokens, skip_special_tokens=True)
		clip_input_tokens = clip.tokenize(vicuna_output, truncate=True).cuda()
		clip_input_feats = self.clip_model.encode_text(clip_input_tokens)
		clip_input_feats = clip_input_feats / clip_input_feats.norm(dim=1, keepdim=True)

		pc_all_porb, _, pc_all_porb_ori, _ = self.classify(clip_input_feats.half(), text_feat, verbose=True)

		pc_all_porb_ori = pc_all_porb_ori[:,:text_num]


		pc_all_porb = list(torch.split(pc_all_porb, box_nums, dim=0))
		pc_all_porb = self.outputs_pad(pc_all_porb, self.nqueries)
		pc_all_porb_ori = list(torch.split(pc_all_porb_ori, box_nums, dim=0))
		pc_all_porb_ori = self.outputs_pad(pc_all_porb_ori, self.nqueries)



		return None, pc_all_porb, pc_all_porb_ori
 
	def classify_scene(self, pc_query_feat):
		box_num = pc_query_feat.shape[0]
		vicuna_pc_feats = self.global_projector(pc_query_feat)
		vicuna_pc_feats = vicuna_pc_feats.unsqueeze(1)
		vicuna_input_base = torch.cat([self.pre_question_embeddings_scene.expand(box_num, -1, -1), vicuna_pc_feats.half(), self.post_question_embeddings_scene.expand(box_num, -1, -1)], dim=1)
		del pc_query_feat
		del vicuna_pc_feats
		gc.collect()
		torch.cuda.empty_cache()
		vicuna_output_tokens = self.vicuna.generate(inputs_embeds=vicuna_input_base, generation_config=self.vicuna_generation_config, max_new_tokens=self.max_new_tokens)
		vicuna_output = self.vicuna_tokenizer.batch_decode(vicuna_output_tokens, skip_special_tokens=True)
		scene = torch.tensor([10] * box_num).cuda()
		for i in range(box_num):
			if vicuna_output[i] == 'bedroom':
				scene[i] = 0
			elif vicuna_output[i] == 'living room': 
				scene[i] = 1
			elif vicuna_output[i] == 'bathroom':
				scene[i] = 2
			elif vicuna_output[i] == 'kitchen':
				scene[i] = 3
			elif vicuna_output[i] == 'conference room':
				scene[i] = 4
			elif vicuna_output[i] == 'warehouse':
				scene[i] = 5
			elif vicuna_output[i] == 'office':
				scene[i] = 6
			elif vicuna_output[i] == 'classroom':
				scene[i] = 7
			elif vicuna_output[i] == 'library':
				scene[i] = 8
			elif vicuna_output[i] == 'dining room':
				scene[i] = 9
		return scene
 
	def outputs_pad(self, output, max_box_num):
		output_dim = output[0].ndim
		batch_size = len(output)
		for i in range(batch_size):
			pad_after = [0,max_box_num-(output[i].shape[0])]
			pad_before = [0] * ((output_dim-1) * 2)
			output[i] = pad(output[i], pad_before + pad_after)
		return torch.stack(output, dim=0)

	def flip_axis_to_depth(self, pc):
		pc2 = np.copy(pc)
		pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]
		pc2[..., 2] *= -1
		return pc2

	@torch.no_grad()
	def pred_bbox_nms(self, outputs, pc_query_feat, point_clouds):
		pc_query_feat = pc_query_feat.permute(1, 0, 2)
		pred_box_corners = outputs['box_corners'].cpu().numpy()
		pred_objectness_prob = outputs['objectness_prob'].cpu().numpy()
		batch_size, query_num = pred_objectness_prob.shape
		new_box_count = []

		sem_cls_logits = []
		center_normalized = []
		center_unnormalized = []
		size_normalized = []
		size_unnormalized = []
		angle_logits = []
		angle_residual = []
		angle_residual_normalized = []
		angle_continuous = []
		objectness_prob = []
		sem_cls_prob = []
		box_corners = []
		pc_query_feats = []
		picks = []

		pred_bboxes = torch.cat([outputs["center_unnormalized"], outputs["size_unnormalized"], torch.unsqueeze(outputs["angle_continuous"], dim=-1)], dim=-1).detach().cpu().numpy()
		pred_bboxes_corners = self.compute_all_bbox_corners(pred_bboxes)

		for bs_ind in range(batch_size):
			pick = []
			for p in range(query_num):
				if outputs['objectness_prob'][bs_ind, p] > 0.10:
					cur_bbox_corner = pred_bboxes_corners[bs_ind, p, :, :]
					if len(extract_pc_in_box3d(point_clouds[bs_ind], cur_bbox_corner)[0]) > 100:
						pick.append(p)
			if pick == []:
				pick.append(torch.argmax(outputs['objectness_prob'][bs_ind]).item())
			boxes_3d_with_prob = []
			for bbox_ind in pick:

				cur_bbox_with_prob = np.zeros(7)
				cur_bbox_corner = pred_box_corners[bs_ind, bbox_ind, :, :]

				cur_bbox_with_prob[0] = np.min(cur_bbox_corner[:, 0])
				cur_bbox_with_prob[1] = np.min(cur_bbox_corner[:, 1])
				cur_bbox_with_prob[2] = np.min(cur_bbox_corner[:, 2])
				cur_bbox_with_prob[3] = np.max(cur_bbox_corner[:, 0])
				cur_bbox_with_prob[4] = np.max(cur_bbox_corner[:, 1])
				cur_bbox_with_prob[5] = np.max(cur_bbox_corner[:, 2])
				cur_bbox_with_prob[6] = pred_objectness_prob[bs_ind, bbox_ind]
				boxes_3d_with_prob.append(cur_bbox_with_prob)

			boxes_3d_with_prob = np.stack(boxes_3d_with_prob, axis=0)
			nms_pick = nms_3d_faster(boxes_3d_with_prob, overlap_threshold = 0.60)
			new_pick = []
			for p in nms_pick:
				new_pick.append(pick[p])
			pick = new_pick
			new_box_count.append(len(pick))

			sem_cls_logits.append(outputs['sem_cls_logits'][bs_ind, pick, :])
			center_normalized.append(outputs['center_normalized'][bs_ind, pick, :])
			center_unnormalized.append(outputs['center_unnormalized'][bs_ind, pick, :])
			size_normalized.append(outputs['size_normalized'][bs_ind, pick, :])
			size_unnormalized.append(outputs['size_unnormalized'][bs_ind, pick, :])
			angle_logits.append(outputs['angle_logits'][bs_ind, pick, :])
			angle_residual.append(outputs['angle_residual'][bs_ind, pick, :])
			angle_residual_normalized.append(outputs['angle_residual_normalized'][bs_ind, pick, :])
			angle_continuous.append(outputs['angle_continuous'][bs_ind, pick])
			objectness_prob.append(outputs['objectness_prob'][bs_ind, pick])
			sem_cls_prob.append(outputs['sem_cls_prob'][bs_ind, pick, :])
			box_corners.append(outputs['box_corners'][bs_ind, pick, :, :])
			pc_query_feats.append(pc_query_feat[bs_ind, pick, :])
			picks.append(pick)

		max_box_num = max(new_box_count)
		new_outputs = {}

		new_outputs['sem_cls_logits'] = self.outputs_pad(sem_cls_logits, self.nqueries)
		new_outputs['center_normalized'] = self.outputs_pad(center_normalized, self.nqueries)
		new_outputs['center_unnormalized'] = self.outputs_pad(center_unnormalized, self.nqueries)
		new_outputs['size_normalized'] = self.outputs_pad(size_normalized, self.nqueries)
		new_outputs['size_unnormalized'] = self.outputs_pad(size_unnormalized, self.nqueries)
		new_outputs['angle_logits'] = self.outputs_pad(angle_logits, self.nqueries)
		new_outputs['angle_residual'] = self.outputs_pad(angle_residual, self.nqueries)
		new_outputs['angle_residual_normalized'] = self.outputs_pad(angle_residual_normalized, self.nqueries)
		new_outputs['angle_continuous'] = self.outputs_pad(angle_continuous, self.nqueries)
		new_outputs['objectness_prob'] = self.outputs_pad(objectness_prob, self.nqueries)
		new_outputs['sem_cls_prob'] = self.outputs_pad(sem_cls_prob, self.nqueries)
		new_outputs['box_corners'] = self.outputs_pad(box_corners, self.nqueries)

		pc_query_feats = torch.cat(pc_query_feats, dim=0)

		return new_outputs, pc_query_feats, new_box_count, picks

	def collect_matched_patch(self, cur_patches, assignments):
		batch_size, query_num, _, _, _ = cur_patches.shape
		all_matched_patch = []

		for cur_bs in range(batch_size):
			cur_assignments = assignments["assignments"][cur_bs]

			cur_matched_cnt = cur_assignments[0].shape[0]

			for cur_match in range(cur_matched_cnt):
				cur_match_patch_ind = cur_assignments[0][cur_match]
				cur_match_gt_ind = cur_assignments[1][cur_match]

				cur_match_patch = cur_patches[cur_bs, cur_match_patch_ind, ...]
				all_matched_patch.append(cur_match_patch)

		all_matched_patch = torch.stack(all_matched_patch, dim=0)
		return all_matched_patch

	def collect_matched_pred_bbox(self, pred_bboxes, assignments):
		batch_size, query_num, _ = pred_bboxes.shape
		all_matched_bbox = []

		for cur_bs in range(batch_size):
			cur_assignments = assignments["assignments"][cur_bs]

			cur_matched_cnt = cur_assignments[0].shape[0]
                          

			for cur_match in range(cur_matched_cnt):
				cur_match_bbox_ind = cur_assignments[0][cur_match]
				cur_match_gt_ind = cur_assignments[1][cur_match]

				cur_match_bbox = pred_bboxes[cur_bs, cur_match_bbox_ind, ...]
				all_matched_bbox.append(cur_match_bbox)

		all_matched_bbox = np.stack(all_matched_bbox, axis=0)
		return all_matched_bbox

	def collect_matched_query_feat(self, pc_query_feat, targets, assignments):
		query_num, batch_size, feat_dim = pc_query_feat.shape

		all_matched_query = []
		all_matched_label = []
		all_matched_prob = []


		for cur_bs in range(batch_size):
			cur_assignments = assignments["assignments"][cur_bs]
			cur_matched_cnt = cur_assignments[0].shape[0]
                          

			for cur_match in range(cur_matched_cnt):
				cur_match_query_ind = cur_assignments[0][cur_match]
				cur_match_gt_ind = cur_assignments[1][cur_match]

				cur_match_query = pc_query_feat[cur_match_query_ind, cur_bs, :]
				cur_match_gt = targets["gt_box_sem_cls_label"][cur_bs, cur_match_gt_ind]

				all_matched_query.append(cur_match_query)
				all_matched_label.append(cur_match_gt)
				all_matched_prob.append(torch.ones(1, device=cur_match_gt.device))

		all_matched_query = torch.stack(all_matched_query, dim=0)
		all_matched_label = torch.stack(all_matched_label, dim=0)
		all_matched_prob = torch.stack(all_matched_prob, dim=0).reshape(-1)

		pc_feat_label={"pc_feat": all_matched_query,
		   "pc_label": all_matched_label,
		   "pc_prob": all_matched_prob,
		   }
		return pc_feat_label


	def train_bin(self, pc_input, criterion, targets):
		with torch.no_grad():
			pc_output = self.pc_model(pc_input)

		batch_size = pc_output["pc_query_feat"].shape[1]
		query_num = pc_output["pc_query_feat"].shape[0]
 

		criterion.eval()
		_, _, assignments = criterion(pc_output, targets, return_assignments=True)

		pc_feat = pc_output["pc_query_feat"].permute(1,0,2)
		pc_feat = pc_feat.reshape(-1, 256)
		pc_feat = Variable(pc_feat, requires_grad=True)
		objectness_prob = self.bin_head(pc_feat)

		tags = []
		for i in range(batch_size):
			tags.append([j in assignments['assignments'][i][0] for j in range(query_num)])
		tags = torch.tensor(tags).cuda()
		for i in range(batch_size):
			for j in range(len(assignments['assignments'][i][0])):
				a = assignments['assignments'][i][0][j]
				b = assignments['assignments'][i][1][j]
				if box3d_iou(pc_output["outputs"]['box_corners'][i,a], targets['gt_box_corners'][i,b])[0] < 0.25:
					tags[i][a] = False
					continue
				for k in range(query_num):
					if box3d_iou(pc_output["outputs"]['box_corners'][i,a], pc_output["outputs"]['box_corners'][i,k])[0] > 0.6:
						tags[i][k] = True
		tags = tags.reshape(-1) * 1

		loss = cross_entropy(objectness_prob, tags, weight=torch.tensor([1.0, 5.0]).cuda())
		return loss

	def process_blip_answer(self, answer):
		for color in self.colors:
			answer = answer.replace(color, '')
		if ' with ' in answer:
			answer = answer.split(' with ')[0]
		if ' of ' in answer:
			answer = answer.split(' of ')[0]
		if ' in ' in answer:
			answer = answer.split(' in ')[0]
		if ' on ' in answer:
			answer = answer.split(' on ')[0]
		if ' that ' in answer:
			answer = answer.split(' that ')[0]
		if ' to ' in answer:
			answer = answer.split(' to ')[0]
		if ' and ' in answer:
			answer = answer.split(' and ')[0]
		return answer.strip()
 
	def get_assignments(self, outputs):
		assignments = []
		batch_size, query_num = outputs['objectness_prob'].shape
		for bs_ind in range(batch_size):
			pick = []
			for p in range(query_num):
				if outputs['objectness_prob'][bs_ind, p] > 0.05:
					pick.append(p)
			if pick == []:
				pick.append(torch.argmax(outputs['objectness_prob'][bs_ind]))
			assignments.append(pick)
		return assignments

	@torch.no_grad()
	def collect_training_data(self, pc_input, img_input, criterion=None, targets=None, training=True):
		self.pc_model.eval()
		pc_output = self.pc_model(pc_input)

		if training:
			criterion.eval()
			_, _, assignments = criterion(pc_output, targets, return_assignments=True)
			patches_num = 0
			for assign in assignments['assignments']:
				patches_num += assign[0].shape[0]
			if patches_num == 0:
				training_data = {'pc_feats': None}
				patches_num = 0
				return training_data, patches_num

			pc_feat_label = self.collect_matched_query_feat(pc_output["pc_query_feat"], targets, assignments)
			pc_feat = pc_feat_label["pc_feat"]
			pc_label = pc_feat_label["pc_label"]
			blip_answers = [self.class2type[i.item()] for i in pc_label]
			training_data = {'pc_feats': pc_feat}

			blip_answers = [blip_answer+'</s>' for blip_answer in blip_answers]   
			if patches_num > 0:
				vicuna_label = self.vicuna_tokenizer(blip_answers, return_tensors='pt', padding=True).input_ids.cuda()  
				vicuna_label = vicuna_label[:,1:]
				training_data['vicuna_label'] = vicuna_label
				training_data['vicuna_label_embeds'] = self.vicuna.base_model.embed_tokens(vicuna_label)
			else:
				training_data['pc_feats'] = None
				patches_num = 0
			return training_data, patches_num

	@torch.no_grad()
	def collect_training_data_global(self, pc_input, targets=None, training=True):
		self.pc_model.eval()
		pc_feat = self.pc_model(pc_input)["enc_features"]

		if training:
			blip_answers = [self.scenes[i.item()]+'</s>' for i in targets['scene']]
			training_data = {'pc_feats': pc_feat}
  
			vicuna_label = self.vicuna_tokenizer(blip_answers, return_tensors='pt', padding=True).input_ids.cuda()  
			vicuna_label = vicuna_label[:,1:]
			training_data['vicuna_label'] = vicuna_label
			training_data['vicuna_label_embeds'] = self.vicuna.base_model.embed_tokens(vicuna_label)
			return training_data
  
	def forward(self, training_data, patches_num, training=True):                    
		if training:
			if patches_num == 0:
                 
				vicuna_pc_feats = self.projector(torch.zeros(patches_num, 256).cuda())
				loss = torch.sum(vicuna_pc_feats) * 0
				return loss
			vicuna_pc_feats = self.projector(training_data['pc_feats'])                      
			vicuna_pc_feats = vicuna_pc_feats.unsqueeze(1)                        
			vicuna_input_base = torch.cat([self.pre_question_embeddings.expand(patches_num, -1, -1), vicuna_pc_feats.half(), self.post_question_embeddings.expand(patches_num, -1, -1)], dim=1)                                
			base_len = vicuna_input_base.shape[1]
			vicuna_input = torch.cat([vicuna_input_base, training_data['vicuna_label_embeds']], dim=1)
			vicuna_output = self.vicuna(inputs_embeds=vicuna_input)['logits'][:, base_len-1:-1, :]                                    
			vicuna_output = vicuna_output.reshape(-1, 32000)
			vicuna_label = training_data['vicuna_label'].reshape(-1)
			mask = (vicuna_label != 0)                             
			loss = torch.sum(cross_entropy(vicuna_output, vicuna_label, reduction='none') * mask) / patches_num
			return loss

	def forward_global(self, training_data, training=True):                    
		if training:
			batch_size = training_data['pc_feats'].shape[0]
			vicuna_pc_feats = self.global_projector(training_data['pc_feats'])                      
			vicuna_pc_feats = vicuna_pc_feats.unsqueeze(1)                        
			vicuna_input_base = torch.cat([self.pre_question_embeddings.expand(batch_size, -1, -1), vicuna_pc_feats.half(), self.post_question_embeddings.expand(batch_size, -1, -1)], dim=1)                                
			base_len = vicuna_input_base.shape[1]
			vicuna_input = torch.cat([vicuna_input_base, training_data['vicuna_label_embeds']], dim=1)
			vicuna_output = self.vicuna(inputs_embeds=vicuna_input)['logits'][:, base_len-1:-1, :]                                    
			vicuna_output = vicuna_output.reshape(-1, 32000)
			vicuna_label = training_data['vicuna_label'].reshape(-1)
			mask = (vicuna_label != 0)                             
			loss = torch.sum(cross_entropy(vicuna_output, vicuna_label, reduction='none') * mask) / batch_size
			return loss

	@torch.no_grad()
	def forward_eval(self, pc_input):
		pc_output = self.pc_model(pc_input)
		pc_feat = pc_output["pc_query_feat"].permute(1,0,2)
		pc_output["outputs"]["objectness_prob"] = torch.nn.functional.softmax(self.bin_head(pc_feat), dim=2)[:,:,1]

		pc_output["outputs"], pc_output["pc_query_feat"], box_nums, picks = self.pred_bbox_nms(pc_output["outputs"], pc_output["pc_query_feat"], pc_input['point_clouds'].cpu())
		eval_text_output = self.eval_text_feats

		_, pc_objectness_prob, pc_sem_cls_prob = self.classify_pc(pc_output["pc_query_feat"], eval_text_output, self.eval_text_num, box_nums)
		pc_output["outputs"]["sem_cls_prob"] = pc_sem_cls_prob
		my_box_nums = torch.tensor(box_nums).cuda()
		my_box_nums = my_box_nums.unsqueeze(1)

		pc_output["outputs"]["scene"] = self.classify_scene(pc_output["enc_features"])
		pc_output["outputs"]["bbox_num"] = my_box_nums
		return pc_output

def build_3detr(args, dataset_config, local_rank):
	pre_encoder = build_preencoder(args)
	encoder = build_encoder(args)
	decoder = build_decoder(args)

	pc_model = Model3DETR(
	 pre_encoder,
	 encoder,
	 decoder,
	 dataset_config,
	 encoder_dim=args.enc_dim,
	 decoder_dim=args.dec_dim,
	 mlp_dropout=args.mlp_dropout,
	 num_queries=args.nqueries,
	 args=args
	)
	sd = torch.load(LOCALIZATION_PATH, map_location=torch.device("cpu"), weights_only=False)
  
	resume_var = {}
	for k in sd['model'].keys():
		if 'pc_model' in k and 'clip_header' not in k:
			resume_var[k[9:]] = sd['model'][k]
	pc_model.load_state_dict(resume_var)

	projector = Projector(256, 4096)
	bin_head = Projector(256, 2)
	global_projector = Projector(256, 4096)

	if args.test_only:
		sd = torch.load(LOCAL_PROJECTOR_PATH, map_location=torch.device("cpu"), weights_only=False) 
		projector.load_state_dict(sd["model"])

		sd = torch.load(BIN_HEAD_PATH, map_location=torch.device("cpu"), weights_only=False) 
		bin_head.load_state_dict(sd["model"]) 

		sd = torch.load(GLOBAL_PROJECTOR_PATH, map_location=torch.device("cpu"), weights_only=False) 
		global_projector.load_state_dict(sd["model"])

	output_processor = BoxProcessor(dataset_config)
 
	model = DETR_3D_2D(args=args, pc_model=pc_model, projector=projector, bin_head=bin_head, global_projector=global_projector, local_rank=local_rank, dataset_config=dataset_config)

	weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
	weight_dict['loss_giou'] = args.giou_loss_coef
	if args.masks:
		weight_dict["loss_mask"] = args.mask_loss_coef
		weight_dict["loss_dice"] = args.dice_loss_coef
                      
	if args.aux_loss:
		aux_weight_dict = {}
		for i in range(args.dec_layers - 1):
			aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
		weight_dict.update(aux_weight_dict)

	losses = ['labels', 'boxes', 'cardinality']
	if args.masks:
		losses += ["masks"]
  
	return model, output_processor
