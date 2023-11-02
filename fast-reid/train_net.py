#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys, setuptools

sys.path.append('.')
sys.path.append("/home/maojingwei/project/LUPerson/pytorch_to_onnx")
from pytorch_to_onnx import net_to_onnx

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
sys.path.append("/home/maojingwei/project")
from sribd_attendance.sribd_face.modules.body_feature.main import cls_body_feature
from sribd_attendance.config import body_feature_config



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.ROOT = "datasets"
    DATASET = "market"
    cfg.merge_from_list(args.opts)
#    cfg.DATALOADER.NUM_INSTANCE becomes 16, the reason is the mgn_R50_moco.yml has _BASE_
    if args.eval_only:
        cfg.DATASETS.KWARGS = f"data_name:{DATASET}"
        cfg.MODEL.WEIGHTS = "pre_models/market.pth"
        cfg.OUTPUT_DIR = "logs/lup_moco/test/${DATASET}"
        cfg.MODEL.DEVICE = "cuda:0"
    else:
        cfg.SOLVER.IMS_PER_BATCH = 8 # should be int times cfg.DATALOADER.NUM_INSTANCE 
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.DATALOADER.NUM_INSTANCE = 4
        cfg.MODEL.BACKBONE.PRETRAIN_PATH = "pre_models/lup_moco_r50.pth"
        cfg.INPUT.DO_AUTOAUG = False
        cfg.TEST.EVAL_PERIOD = 60
        SPLIT = "id"
        RATIO = "1.0"
        cfg.DATASETS.KWARGS = "data_name:{}+split_mode:{}+split_ratio:{}".format(DATASET, SPLIT, RATIO)
        cfg.OUTPUT_DIR = "logs/lup_moco_r50/{}/{}_{}".format(DATASET, SPLIT, RATIO)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
#        net_to_onnx(model, "lup_model.onnx", [(1,3,384,128),], ["nchw_rgb",], ["output", ], True)

        body_feature_obj = cls_body_feature(body_feature_config["LUPerson"])
        res = DefaultTrainer.test(cfg, model, body_feature_obj=body_feature_obj)
        # cfg.DATASETS.TESTS is ('CMDM', ), means Cuhk Market Duke Msmt17
        return res

    trainer = DefaultTrainer(cfg)
    # load trained model to funetune
    if args.finetune: Checkpointer(trainer.model).load(cfg.MODEL.WEIGHTS)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(args)
    args.config_file = "configs/CMDM/mgn_R50_moco.yml"
    args.eval_only = True
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
