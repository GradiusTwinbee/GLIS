                                                                      
\
\
   
import torch
import torch.nn.functional as F
from torch import nn

from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
        accuracy, get_world_size, interpolate,
        is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
         dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from .position_encoding import PositionEmbeddingSine


class DETR(nn.Module):
                                                               
	def __init__(self, backbone, encoder, decoder, d_model, num_classes, num_queries, cls_head, aux_loss=False):
\
\
\
\
\
\
\
\
     
		super().__init__()
		self.num_queries = num_queries
		self.encoder = encoder
		self.decoder = decoder
                                 
		hidden_dim = d_model
  
                                  
                   
		self.imagenet_embed = nn.Linear(hidden_dim, 1020)
		self.class_embed = cls_head
  
		self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
  
               
		self.query_embed = nn.Embedding(num_queries, hidden_dim)
  
		self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
		self.backbone = backbone
		self.aux_loss = aux_loss
		self.hidden_dim = hidden_dim
  
                                 
		self.clip_header = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
		nn.init.normal_(self.clip_header, std=self.hidden_dim ** -0.5)

	def forward(self, samples: NestedTensor, use_1k_header=False):
\
\
\
\
\
\
\
\
\
\
\
\
\
     
		if isinstance(samples, (list, torch.Tensor)):
			samples = nested_tensor_from_tensor_list(samples)
		features, pos = self.backbone(samples)

		src, mask = features[-1].decompose()
		assert mask is not None
  
                                
		bs = src.shape[0]
		src = self.input_proj(src)
		src = src.flatten(2).permute(2, 0, 1)
		mask = mask.flatten(1)
		pos[-1] = pos[-1].flatten(2).permute(2, 0, 1)
  
                                     
		query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
  
                         
		xyz, memory, xyz_inds = self.encoder(src = src, src_key_padding_mask = mask, pos = pos[-1])
		tgt = torch.zeros_like(query_embed)
		hs, _ = self.decoder(tgt, memory, memory_key_padding_mask = mask, pos=pos[-1], query_pos = query_embed)
		hs = hs.permute(0, 2, 1, 3)
  
                                                                                         
		num_layers, batch, num_queries, channel  = (
		 hs.shape[0],
		 hs.shape[1],
		 hs.shape[2],
		 hs.shape[3],
		)

		if use_1k_header:
			ImageNet_pred = self.imagenet_embed(hs)
  
		class_head_hs = hs.permute(0,1,3,2).reshape(num_layers*batch, channel, num_queries)
		outputs_class = self.class_embed(class_head_hs).permute(0,2,1).reshape(num_layers, batch, num_queries, -1)
  
		outputs_coord = self.bbox_embed(hs).sigmoid()
  
                  
		img_query_feat = hs[-1,:,:,:].permute(1,0,2)
		img_query_feat = img_query_feat @ self.clip_header
		img_query_feat = img_query_feat / img_query_feat.norm(dim=2, keepdim=True)

		out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'img_query_feat': img_query_feat}
                                                                            
  
		if use_1k_header:
			out['imagenet_logits'] = ImageNet_pred[-1]
   
		if self.aux_loss:
			out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
		return out

	@torch.jit.unused
	def _set_aux_loss(self, outputs_class, outputs_coord):
                                                                  
                                                                
                                              
		return [{'pred_logits': a, 'pred_boxes': b}
		  for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(nn.Module):
\
\
\
\
    
	def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
\
\
\
\
\
\
\
     
		super().__init__()
		self.num_classes = num_classes
		self.matcher = matcher
		self.weight_dict = weight_dict
		self.eos_coef = eos_coef
		self.losses = losses
		empty_weight = torch.ones(self.num_classes + 1)
		empty_weight[-1] = self.eos_coef
		self.register_buffer('empty_weight', empty_weight)

	def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
\
\
     
		assert 'pred_logits' in outputs
		src_logits = outputs['pred_logits']

		idx = self._get_src_permutation_idx(indices)
		target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
		target_classes = torch.full(src_logits.shape[:2], self.num_classes,
		       dtype=torch.int64, device=src_logits.device)
		target_classes[idx] = target_classes_o

		loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
		losses = {'loss_ce': loss_ce}

		if log:
                                                                              
			losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
		return losses

	@torch.no_grad()
	def loss_cardinality(self, outputs, targets, indices, num_boxes):
\
\
     
		pred_logits = outputs['pred_logits']
		device = pred_logits.device
		tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
                                                                                      
		card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
		card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
		losses = {'cardinality_error': card_err}
		return losses

	def loss_boxes(self, outputs, targets, indices, num_boxes):
\
\
\
     
		assert 'pred_boxes' in outputs
		idx = self._get_src_permutation_idx(indices)
		src_boxes = outputs['pred_boxes'][idx]
		target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

		loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

		losses = {}
		losses['loss_bbox'] = loss_bbox.sum() / num_boxes

		loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
		 box_ops.box_cxcywh_to_xyxy(src_boxes),
		 box_ops.box_cxcywh_to_xyxy(target_boxes)))
		losses['loss_giou'] = loss_giou.sum() / num_boxes
		return losses

	def loss_masks(self, outputs, targets, indices, num_boxes):
\
\
     
		assert "pred_masks" in outputs

		src_idx = self._get_src_permutation_idx(indices)
		tgt_idx = self._get_tgt_permutation_idx(indices)
		src_masks = outputs["pred_masks"]
		src_masks = src_masks[src_idx]
		masks = [t["masks"] for t in targets]
                                                               
		target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
		target_masks = target_masks.to(src_masks)
		target_masks = target_masks[tgt_idx]

                                           
		src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
		      mode="bilinear", align_corners=False)
		src_masks = src_masks[:, 0].flatten(1)

		target_masks = target_masks.flatten(1)
		target_masks = target_masks.view(src_masks.shape)
		losses = {
		 "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
		 "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
		}
		return losses

	def _get_src_permutation_idx(self, indices):
                                         
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
                                     
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
		loss_map = {
		 'labels': self.loss_labels,
		 'cardinality': self.loss_cardinality,
		 'boxes': self.loss_boxes,
		 'masks': self.loss_masks
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

	def forward(self, outputs, targets):
\
\
\
\
\
     
		outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

                                                                               
		indices = self.matcher(outputs_without_aux, targets)

                                                                                            
		num_boxes = sum(len(t["labels"]) for t in targets)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
		if is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_boxes)
		num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

                                    
		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

                                                                                                   
		if 'aux_outputs' in outputs:
			for i, aux_outputs in enumerate(outputs['aux_outputs']):
				indices = self.matcher(aux_outputs, targets)
				for loss in self.losses:
					if loss == 'masks':
                                                                            
						continue
					kwargs = {}
					if loss == 'labels':
                                                  
						kwargs = {'log': False}
					l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
					l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
					losses.update(l_dict)

		return losses


class PostProcess(nn.Module):
                                                                                        
	@torch.no_grad()
	def forward(self, outputs, target_sizes):
\
\
\
\
\
\
     
		out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

		assert len(out_logits) == len(target_sizes)
		assert target_sizes.shape[1] == 2

		prob = F.softmax(out_logits, -1)
		scores, labels = prob[..., :-1].max(-1)

                                      
		boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
                                                                
		img_h, img_w = target_sizes.unbind(1)
		scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
		boxes = boxes * scale_fct[:, None, :]

		results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

		return results


class MLP(nn.Module):
                                                            

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
		return x

class Projector(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
                                                  
		self.linear = nn.Linear(input_dim, output_dim)
                         
                           

	def forward(self, x):
		s = self.linear(x)
		return s
 
class Global_Projector(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1 = nn.Linear(2048, 1)
		self.linear2 = nn.Linear(256, 4096)

	def forward(self, x):
		x = x.transpose(1, 2)
		x = self.linear1(x)
		x = x.squeeze()
		s = self.linear2(x)
		return s


def build(args):
                                                        
                                                              
                                                              
                                                                  
                                                                       
                                                          
                                                           
                                                                             
	num_classes = 20 if args.dataset_file != 'coco' else 91
	if args.dataset_file == "coco_panoptic":
                                                                        
                                                             
		num_classes = 250
	device = torch.device(args.device)

	backbone = build_backbone(args)

	transformer = build_transformer(args)

	model = DETR(
	 backbone,
	 transformer,
	 num_classes=num_classes,
	 num_queries=args.num_queries,
	 aux_loss=args.aux_loss,
	)
	if args.masks:
		model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
	matcher = build_matcher(args)
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
	criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
	       eos_coef=args.eos_coef, losses=losses)
	criterion.to(device)
	postprocessors = {'bbox': PostProcess()}
	if args.masks:
		postprocessors['segm'] = PostProcessSegm()
		if args.dataset_file == "coco_panoptic":
			is_thing_map = {i: i <= 90 for i in range(201)}
			postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

	return model, criterion, postprocessors
