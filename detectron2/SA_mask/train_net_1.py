#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
SA_Mask Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator


#from tridentnet import add_tridentnet_config ### New
import sa_mask.sa_mask_rcnn
import sa_mask.sa_mask_rpn
import sa_mask.sa_mask_fpn
from sa_mask.config import get_cfg




from detectron2.data.datasets import register_coco_instances ### New
from detectron2.data import MetadataCatalog, DatasetCatalog ### New



class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    ### add_tridentnet_config(cfg) ### New
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    
    register_coco_instances("nuclei_HE_train", {}, "/tmp2/chacotw/data/nuclei_instances_fss_cell_val.json", "/tmp2/chacotw/data/")
    register_coco_instances("nuclei_HE_val", {},"/tmp2/chacotw/data/nuclei_instances_fss_cell_val.json", "/tmp2/chacotw/data/")
    register_coco_instances("nuclei_HE_test", {}, "/tmp2/chacotw/data/nuclei_instances_fss_cell_test.json", "/tmp2/chacotw/data/")
    
    nuclei_HE_train_metadata = MetadataCatalog.get("nuclei_HE_train")
    dataset_dicts = DatasetCatalog.get("nuclei_HE_train")

    cfg.DATASETS.TRAIN = ("nuclei_HE_train",)
    cfg.DATASETS.TEST = ("nuclei_HE_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
