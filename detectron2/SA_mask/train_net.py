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
import sa_mask.modeling.sa_mask_rcnn
import sa_mask.modeling.sa_mask_rpn
import sa_mask.modeling.sa_mask_fpn
from sa_mask.config import get_cfg
from sa_mask.data.coco import register_coco_instances ### New


from detectron2.data import MetadataCatalog, DatasetCatalog ### New


"""
Test with ballon dataset
"""
from detectron2.structures import BoxMode
import json, cv2
import numpy as np

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


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
    
    
    register_coco_instances("nuclei_HE_train", {}, "/tmp2/chacotw/data/nuclei_instances_fss_cell_val.json", "/tmp2/chacotw/data/", use_sa=True)
    register_coco_instances("nuclei_HE_val", {},"/tmp2/chacotw/data/nuclei_instances_fss_cell_val.json", "/tmp2/chacotw/data/", use_sa=True)
    register_coco_instances("nuclei_HE_test", {}, "/tmp2/chacotw/data/nuclei_instances_fss_cell_test.json", "/tmp2/chacotw/data/", use_sa=True)
    
    nuclei_HE_train_metadata = MetadataCatalog.get("nuclei_HE_train")
    dataset_dicts = DatasetCatalog.get("nuclei_HE_train")
    
    """
    Test with ballon dataset
    """

#     for d in ["train", "val"]:
#         DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("../balloon/" + d))
#         MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
#     balloon_metadata = MetadataCatalog.get("balloon_train")
    
    
#     cfg.DATASETS.TRAIN = ("balloon_train",)
    
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
