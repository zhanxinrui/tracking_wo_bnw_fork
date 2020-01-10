import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# class FRCNN_FPN(FasterRCNN):
class MRCNN_FPN(MaskRCNN):
    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(MRCNN_FPN, self).__init__(backbone, num_classes)
        # print(self)
        # in_features = self.roi_heads.box_predictor.cls_score.in_features  # 1024
        # self.roi_heads.box_predictor =  FastRCNNPredictor(in_features, num_classes)

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def predict_boxes(self, images, boxes):
        device = list(self.parameters())[0].device # to cuda
        images = images.to(device)
        boxes = boxes.to(device)#boxes.size() = nx4
        # masks =  masks.to(device)#!！新加一个mask,需要提前得到mask或者现在运行一遍模型得到。。　类似MOT给出的result
        targets = None
        original_image_sizes = [img.shape[-2:] for img in images]#1920x1080

        images, targets = self.transform(images, targets)#归一化处理  targets没有变化还是none，为什么

        features = self.backbone(images.tensors)#骨干网络得到特征5种尺度  features[0].size() = [1,256,192,336]
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        # proposals, proposal_losses = self.rpn(images, features, targets)
        from torchvision.models.detection.transform import resize_boxes
        boxes = resize_boxes(
            boxes, original_image_sizes[0], images.image_sizes[0])#去掉黑边是749*1333因为是不同相机，但是得把他放到固定的画幅上
        proposals = [boxes]
        #mask不具备回归的能力
        box_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)# box_features.size() = [n,256,7,7],
        box_features = self.roi_heads.box_head(box_features)#box_features.size() = [n,1024] 1024个输出，
        class_logits, box_regression = self.roi_heads.box_predictor(#
            box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(
            pred_boxes, images.image_sizes[0], original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def load_image(self, img):
        pass
