                                                  
from .model_3detr import build_3detr

MODEL_FUNCS = {
    "3detr": build_3detr,
}

def build_model(args, dataset_config, local_rank):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config, local_rank)
    return model, processor