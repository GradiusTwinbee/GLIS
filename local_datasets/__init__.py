                                                  
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    dataset_dict = {
        "train": dataset_builder(dataset_config, split_set="train", root_dir=args.dataset_root_dir, augment=True, args=args),
        "test": dataset_builder(dataset_config, split_set="val", root_dir=args.dataset_root_dir, augment=False, args=args),
    }
    return dataset_dict, dataset_config
    
