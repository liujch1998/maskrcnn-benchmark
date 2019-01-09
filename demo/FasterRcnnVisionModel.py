import torch
import torch.nn as nn

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

class FasterRcnnVisionModel (nn.Module):

    def __init__ (self):
        super(FasterRcnnBackbone, self).__init__()

        config_file = '../configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml'
        cfg.merge_from_file(config_file)
        coco_demo = COCODemo(cfg)

    def forward (self, image):
        '''
        > image cv2image
        < region_visual_feats [(d_region_visual_feats=1024) * n_regions=1000]
        < bboxes [[float * 4] * n_regions=1000]
        < image_feats (1, d_image_feats)
        '''

        bbox_feats, bboxes_reg = coco_demo.compute_prediction(image)
        region_visual_feats = list(bbox_feats.cpu().chunk(bbox_feats.size(0), dim=0))
        bboxes = bboxes_reg.bbox.tolist()
        image_feats = torch.tensor([])
        return region_visual_feats, bboxes, image_feats
