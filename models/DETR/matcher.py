                                                                      
\
\
   
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
\
\
\
\
\
       

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
\
\
\
\
\
\
           
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
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
\
\
\
           
        bs, num_queries = outputs["pred_logits"].shape[:2]

                                                            
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)                                           
        out_bbox = outputs["pred_boxes"].flatten(0, 1)                                 

                                                 
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

                                                                                      
                                                        
                                                                                   
        cost_class = -out_prob[:, tgt_ids]

                                           
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

                                            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

                           
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
