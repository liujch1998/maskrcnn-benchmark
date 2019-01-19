import torch
import torch.nn as nn

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

class FasterRcnnVisionModel (nn.Module):

    def __init__ (self):
        super(FasterRcnnVisionModel, self).__init__()

        config_file = '/home/jl25/research/maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml'
        cfg.merge_from_file(config_file)
        self.coco_demo = COCODemo(cfg)

    def forward (self, image):
        '''
        > image cv2image
        < bboxes [[float * 4] * n_regions=1000]
        < regions_visual_feats [(d_region_visual_feats=1024) * n_regions=1000]
        < objectnesses [float * n_regions=1000]
        < image_feats (1, d_image_feats)
        '''

        bbox_feats, bboxes_reg, objectness = self.coco_demo.compute_prediction(image)
        bboxes = bboxes_reg.bbox.tolist()[:-1]
        bboxes_feats = list(bbox_feats.cpu().chunk(bbox_feats.size(0), dim=0))
        regions_visual_feats = bboxes_feats[:-1]
        image_feats = bbox_feats[-1,:].unsqueeze(0).cpu().clone().detach()
        objectnesses = objectness.tolist()
        return bboxes, regions_visual_feats, objectnesses, image_feats
