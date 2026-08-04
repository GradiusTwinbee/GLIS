                                                  
 
                                                                 
                                                         

\
   
import logging
import os
import sys
from collections import OrderedDict
import pickle as pk

import numpy as np
import scipy.special as scipy_special
import torch

from utils.box_util import (extract_pc_in_box3d, flip_axis_to_camera_np,
                            get_3d_box, get_3d_box_batch, box3d_iou)
from utils.eval_det import eval_det_multiprocessing, get_iou_obb
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls

from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from config import SCENE2CLASS

POSTPROCESS_SCORE_SEM_EXP = 1.0
POSTPROCESS_SCORE_OBJ_EXP = 1.0
POSTPROCESS_TOPK = 0
POSTPROCESS_NMS_IOU = 0.25
POSTPROCESS_CONF_THRESH = 0
POSTPROCESS_MIN_BOX_EDGE = 0.03
POSTPROCESS_MAX_BOX_EDGE = 0
POSTPROCESS_MIN_BOX_VOLUME = 0.01
POSTPROCESS_MAX_BOX_VOLUME = 0
POSTPROCESS_MIN_BOX_HEIGHT = 0.035
POSTPROCESS_MIN_BOX_BASE_AREA = 0.045

def takeThird(elem):
    return elem[2]


def get_box_dims_from_corners(corners):
    edge_a = np.linalg.norm(corners[:, :, 0, :] - corners[:, :, 1, :], axis=-1)
    edge_b = np.linalg.norm(corners[:, :, 1, :] - corners[:, :, 2, :], axis=-1)
    edge_c = np.linalg.norm(corners[:, :, 0, :] - corners[:, :, 4, :], axis=-1)
    return np.stack([edge_a, edge_b, edge_c], axis=-1)

def get_box_height_base_area_from_corners(corners):
    box_height = np.abs(corners[:, :, 0, 1] - corners[:, :, 4, 1])
    box_dims = get_box_dims_from_corners(corners)
    box_base_area = box_dims[:, :, 0] * box_dims[:, :, 1]
    return box_height, box_base_area

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]                            
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
                                    
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


                                                                         
def parse_predictions(
    predicted_boxes, sem_cls_probs, objectness_probs, point_cloud, config_dict, objs
):
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
\
\
       

    sem_cls_probs = sem_cls_probs.detach().cpu().numpy()                     
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)                  
    pred_sem_cls = np.argmax(sem_cls_probs, -1)
    obj_prob = objectness_probs.detach().cpu().numpy()
    score_sem_exp = config_dict.get("score_sem_exp", 1.0)
    score_obj_exp = config_dict.get("score_obj_exp", 1.0)
    topk = config_dict.get("topk", 0)
    min_box_edge = config_dict.get("min_box_edge", 0)
    max_box_edge = config_dict.get("max_box_edge", 0)
    min_box_volume = config_dict.get("min_box_volume", 0)
    max_box_volume = config_dict.get("max_box_volume", 0)
    min_box_height = config_dict.get("min_box_height", 0)
    min_box_base_area = config_dict.get("min_box_base_area", 0)

    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy()

    K = pred_corners_3d_upright_camera.shape[1]                   
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K))

    if (
        min_box_edge > 0
        or max_box_edge > 0
        or min_box_volume > 0
        or max_box_volume > 0
        or min_box_height > 0
        or min_box_base_area > 0
    ):
        box_dims = get_box_dims_from_corners(pred_corners_3d_upright_camera)
        box_min_edge = np.min(box_dims, axis=-1)
        box_max_edge = np.max(box_dims, axis=-1)
        box_volume = np.prod(box_dims, axis=-1)
        size_mask = np.ones((bsize, K), dtype=bool)
        if min_box_edge > 0:
            size_mask &= box_min_edge >= min_box_edge
        if max_box_edge > 0:
            size_mask &= box_max_edge <= max_box_edge
        if min_box_volume > 0:
            size_mask &= box_volume >= min_box_volume
        if max_box_volume > 0:
            size_mask &= box_volume <= max_box_volume
        if min_box_height > 0 or min_box_base_area > 0:
            box_height, box_base_area = get_box_height_base_area_from_corners(
                pred_corners_3d_upright_camera
            )
            if min_box_height > 0:
                size_mask &= box_height >= min_box_height
            if min_box_base_area > 0:
                size_mask &= box_base_area >= min_box_base_area
        nonempty_box_mask *= size_mask
        for i in range(bsize):
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1

    if config_dict["remove_empty_box"]:
                                               
                                                                
        batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]         
        for i in range(bsize):
            pc = batch_pc[i, :, :]         
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]         
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1
                                               

    if "no_nms" in config_dict and config_dict["no_nms"]:
                                         
        pred_mask = nonempty_box_mask
    elif not config_dict["use_3d_nms"]:
                                                                     
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_2d_faster(
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
                                                               
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
                                                                     
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
                                                               
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
                                                                     
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]                                                          
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
                                                               

    batch_pred_map_cls = (
        []
    )                                                                                                                        
    for i in range(bsize):
        if config_dict["per_class_proposal"]:
            assert config_dict["use_cls_confidence_only"] is False
            cur_list = []
            valid_proposal_mask = (pred_mask[i, :] == 1) & (
                obj_prob[i, :] > config_dict["conf_thresh"]
            )
            if topk > 0 and valid_proposal_mask.sum() > topk:
                allowed_objs = objs[i]
                proposal_scores = np.zeros(pred_corners_3d_upright_camera.shape[1])
                if len(allowed_objs) > 0:
                    proposal_scores = (
                        np.max(sem_cls_probs[i][:, allowed_objs] ** score_sem_exp, axis=1)
                        * (obj_prob[i, :] ** score_obj_exp)
                    )
                valid_inds = np.where(valid_proposal_mask)[0]
                topk_inds = valid_inds[
                    np.argsort(-proposal_scores[valid_inds])[:topk]
                ]
                topk_mask = np.zeros_like(valid_proposal_mask)
                topk_mask[topk_inds] = True
                valid_proposal_mask = topk_mask
            for ii in range(config_dict["dataset_config"].num_semcls):
                temp_list = [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        (sem_cls_probs[i, j, ii] ** score_sem_exp)
                        * (obj_prob[i, j] ** score_obj_exp),
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if valid_proposal_mask[j]
                    and ii in objs[i]
                ]
                cur_list += temp_list
            batch_pred_map_cls.append(cur_list)
        elif config_dict["use_cls_confidence_only"]:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, pred_sem_cls[i, j].item()],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                    and pred_sem_cls[i, j].item() in objs[i]
                ]
            )

    return batch_pred_map_cls


def get_ap_config_dict(
    remove_empty_box=False,
    use_3d_nms=True,
    nms_iou=POSTPROCESS_NMS_IOU,
    use_old_type_nms=False,
    cls_nms=True,
    per_class_proposal=True,
    use_cls_confidence_only=False,
    conf_thresh=POSTPROCESS_CONF_THRESH,
    no_nms=False,
    dataset_config=None,
    score_sem_exp=POSTPROCESS_SCORE_SEM_EXP,
    score_obj_exp=POSTPROCESS_SCORE_OBJ_EXP,
    topk=POSTPROCESS_TOPK,
    min_box_edge=POSTPROCESS_MIN_BOX_EDGE,
    max_box_edge=POSTPROCESS_MAX_BOX_EDGE,
    min_box_volume=POSTPROCESS_MIN_BOX_VOLUME,
    max_box_volume=POSTPROCESS_MAX_BOX_VOLUME,
    min_box_height=POSTPROCESS_MIN_BOX_HEIGHT,
    min_box_base_area=POSTPROCESS_MIN_BOX_BASE_AREA,
):
\
\
       

    config_dict = {
        "remove_empty_box": remove_empty_box,
        "use_3d_nms": use_3d_nms,
        "nms_iou": nms_iou,
        "use_old_type_nms": use_old_type_nms,
        "cls_nms": cls_nms,
        "per_class_proposal": per_class_proposal,
        "use_cls_confidence_only": use_cls_confidence_only,
        "conf_thresh": conf_thresh,
        "no_nms": no_nms,
        "dataset_config": dataset_config,
        "score_sem_exp": score_sem_exp,
        "score_obj_exp": score_obj_exp,
        "topk": topk,
        "min_box_edge": min_box_edge,
        "max_box_edge": max_box_edge,
        "min_box_volume": min_box_volume,
        "max_box_volume": max_box_volume,
        "min_box_height": min_box_height,
        "min_box_base_area": min_box_base_area,
    }
    return config_dict


class APCalculator(object):
                                       

    def __init__(
        self,
        dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=None,
        exact_eval=True,
        ap_config_dict=None,
    ):
\
\
\
\
\
           
        dataset_config.num_semcls = 20
        self.ap_iou_thresh = ap_iou_thresh
        if ap_config_dict is None:
            ap_config_dict = get_ap_config_dict(
                dataset_config=dataset_config, remove_empty_box=exact_eval
            )
        self.ap_config_dict = ap_config_dict
        self.class2type_map = class2type_map
        self.scene2class = pk.load(open(SCENE2CLASS,'rb'))
        self.scene_gt_pred = np.zeros([11, 11], dtype=int)
        self.pred_scene_gt_obj = np.zeros([11, 20], dtype=int)
        self.reset()

    def make_gt_list(self, gt_box_corners, gt_box_sem_cls_labels, gt_box_present):
        batch_gt_map_cls = []
        bsize = gt_box_corners.shape[0]
        for i in range(bsize):
            batch_gt_map_cls.append(
                [
                    (gt_box_sem_cls_labels[i, j].item(), gt_box_corners[i, j])
                    for j in range(gt_box_corners.shape[1])
                    if gt_box_present[i, j] == 1
                ]
            )
        return batch_gt_map_cls

    def step_meter(self, outputs, targets):
        if "outputs" in outputs:
            outputs = outputs["outputs"]
        bs = len(outputs["scene"])
        for i in range(bs):
            self.scene_gt_pred[targets["scene"][i]][outputs["scene"][i]] += 1
            for j in range(targets["bbox_num"][i]):
                self.pred_scene_gt_obj[outputs["scene"][i]][targets["gt_box_sem_cls_label"][i][j]] += 1
        self.step(
            predicted_box_corners=outputs["box_corners"],
            sem_cls_probs=outputs["sem_cls_prob"],
            objectness_probs=outputs["objectness_prob"],
            point_cloud=targets["point_clouds"],
            gt_box_corners=targets["gt_box_corners"],
            gt_box_sem_cls_labels=targets["gt_box_sem_cls_label"],
            gt_box_present=targets["gt_box_present"],
            objs=[self.scene2class[scene.item()] for scene in outputs["scene"]],
        )

    def step(
        self,
        predicted_box_corners,
        sem_cls_probs,
        objectness_probs,
        point_cloud,
        gt_box_corners,
        gt_box_sem_cls_labels,
        gt_box_present,
        objs,
    ):
\
\
\
           
        gt_box_corners = gt_box_corners.cpu().detach().numpy()
        gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
        gt_box_present = gt_box_present.cpu().detach().numpy()
        batch_gt_map_cls = self.make_gt_list(
            gt_box_corners, gt_box_sem_cls_labels, gt_box_present
        )

        batch_pred_map_cls = parse_predictions(
            predicted_box_corners,
            sem_cls_probs,
            objectness_probs,
            point_cloud,
            self.ap_config_dict,
            objs
        )

        self.accumulate(batch_pred_map_cls, batch_gt_map_cls)

    def accumulate(self, batch_pred_map_cls, batch_gt_map_cls):
\
\
\
\
\
\
           
        bsize = len(batch_pred_map_cls)
        assert bsize == len(batch_gt_map_cls)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
                                                                                        
        overall_ret = OrderedDict()
        for ap_iou_thresh in self.ap_iou_thresh:
            ret_dict = OrderedDict()
            rec, prec, ap = eval_det_multiprocessing(
                self.pred_map_cls, self.gt_map_cls, ovthresh=ap_iou_thresh
            )
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[key] if key in self.class2type_map.keys() else str(key)
                ret_dict["%s Average Precision" % (clsname)] = ap[key]
            ap_vals = np.array(list(ap.values()), dtype=np.float32)
            ap_vals[np.isnan(ap_vals)] = 0
            ret_dict["mAP"] = ap_vals.mean()
            rec_list = []
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[key] if key in self.class2type_map.keys() else str(key)
                try:
                    ret_dict["%s Recall" % (clsname)] = rec[key][-1]
                    rec_list.append(rec[key][-1])
                except:
                    ret_dict["%s Recall" % (clsname)] = 0
                    rec_list.append(0)
            ret_dict["AR"] = np.mean(rec_list)
            overall_ret[ap_iou_thresh] = ret_dict
        return overall_ret

    def __str__(self):
        overall_ret = self.compute_metrics()
        return self.metrics_to_str(overall_ret)

    def metrics_to_str(self, overall_ret, per_class=True):
        mAP_strs = []
        AR_strs = []
        per_class_metrics = []
        for ap_iou_thresh in self.ap_iou_thresh:
            mAP = overall_ret[ap_iou_thresh]["mAP"] * 100
            mAP_strs.append(f"{mAP:.2f}")
            ar = overall_ret[ap_iou_thresh]["AR"] * 100
            AR_strs.append(f"{ar:.2f}")

            if per_class:
                                   
                per_class_metrics.append("-" * 5)
                per_class_metrics.append(f"IOU Thresh={ap_iou_thresh}")
                for x in list(overall_ret[ap_iou_thresh].keys()):
                    if x == "mAP" or x == "AR":
                        pass
                    else:
                        met_str = f"{x}: {overall_ret[ap_iou_thresh][x]*100:.2f}"
                        per_class_metrics.append(met_str)

        ap_header = [f"mAP{x:.2f}" for x in self.ap_iou_thresh]
        ap_str = ", ".join(ap_header)
        ap_str += ": " + ", ".join(mAP_strs)
        ap_str += "\n"

        ar_header = [f"AR{x:.2f}" for x in self.ap_iou_thresh]
        ap_str += ", ".join(ar_header)
        ap_str += ": " + ", ".join(AR_strs)

        if per_class:
            per_class_metrics = "\n".join(per_class_metrics)
            ap_str += "\n"
            ap_str += per_class_metrics

        return ap_str

    def metrics_to_dict(self, overall_ret):
        metrics_dict = {}
        for ap_iou_thresh in self.ap_iou_thresh:
            metrics_dict[f"mAP_{ap_iou_thresh}"] = (
                overall_ret[ap_iou_thresh]["mAP"] * 100
            )
            metrics_dict[f"AR_{ap_iou_thresh}"] = overall_ret[ap_iou_thresh]["AR"] * 100
        return metrics_dict

    def reset(self):
        self.gt_map_cls = {}                                  
        self.pred_map_cls = {}                                         
        self.scan_cnt = 0
