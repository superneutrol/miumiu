# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from Model import detr , Position_encoding, backbone, segmentation , transformer 

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss
from Model.backbone import Joiner
from detr import DETR, SetCriterion
from Model.matcher import HungarianMatcher
from Model.Position_encoding import PositionEmbeddingSine
from Model.transformer import Transformer
from Model.segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from Utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from Utils.misc import NestedTensor
from datasets.coco import convert_coco_poly_to_mask

# khởi tạo danh sách __all__ lưu trữ một chuỗi là DETr 

__all__ = ["Detr"]

# Khởi tạo một lớp MaskedBackbone để thực hiện 
# tạo một lớp mòng xung quanh backbone của D2 để cung cấp các lớp đệm 
class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính tham số 
    def __init__(self, cfg):
        # super.init 
        super().__init__()
        # backbone 
        self.backbone = build_backbone(cfg) # truyền vào mô hình các tham số 
        # backbone output 
        backbone_shape = self.backbone.output_shape()
        # danh sách các strides của features map trong backbone 
        self.features_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        # số lượng kênh của features map lớp backbone cuối cùng 
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    # Thiết lập phương thức forward , phương thức này sẽ được gọi khi các 
    # tham số được chuyền vào 
    def forward(self, images):
        # Tính toán các đặc trưng từ backbone cho các hình ảnh 
        # kết qủa là nhận được các featres map 
        features = self.backbone(images.tensor)
        # thực hiện đệm các mặt nạ để loại bỏ padding từ các features map 
        # sử dụng hàm mask_out_padding se trả về một danh sách các mặt nạ , mỗi mặt nạ tương ứng với 
        # một features map và chỉ ra những vùng nào là padding 
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        # đảm bảo rằng số lượng features map bằng số lượng masks 
        assert len(features) == len(masks)
        # duyệt qua dnah sách các features trong features 
        for i , k in enumerate(features.keys()):
            # chuyển đổi danh sách features theo giá trị tensor k 
            # thành các nested Tensor để lưu trữ các features map tensor có kích thước không đồng nhất 
            # là các features có kích thước padding khác nhau 
            features[k] = NestedTensor(features[k], masks[i])
        # trả về danh sách features với các biểu diễn features map là NestedTensor 
        return features 
    
    # Thiết  lập phương thức mask_out_padding thực đệm các mặt nạ để 
    def mask_out_padding(self, feature_shapes, image_sizes, device):
        # khởi tạo một danh sách mask để lưu trữ các masks sau khi đêmh chúng 
        masks = []
        # đảm bảo rằng len(feature_shapes) = độ dài của danh sách fearures strides
        # tức là các features map sẽ khớp với số lượng các cửa sổ trượt 
        assert len(feature_shapes) == len(self.features_strides)
        # duyệt qua danh sách features_shape là kích thước đầu ra của backbone 
        # lấy ra chỉ số và các giá trị của tensor này 
        for idx, shape in enumerate(feature_shapes):
            # lấy ra các kích thước của shape 
            N, _, H, W = shape 
            # tạo một tensor masks_per_feature_leval có shape = [N, H, W]
            # là tensor boolean được lấp đầy = True 
            masks_per_feature_level = torch.ones(size=(N, H, W), dtype=torch.bool, device=device)
            # duyệt qua danh sách image_size lấy ra các chỉ số và kích thước của mỗi image 
            for image_idx, (h, w) in enumerate(image_sizes):
                # Thực hiện việc đệm các features map 
                masks_per_feature_level[ # danh sách này sẽ chứa các kết quả đệm cho mỗi features map 
                    # nhận đầu vào là chỉ số của hình ảnh 
                    image_idx, 
                    # Tính toán tỷ lệ cần đệm theo 2 chiều h và w dựa trên kích thước của cửa sổ trượt 
                    # tương ứng 
                    : int(np.ceil(float(h) / self.features_strides[idx])),
                    # hàm np.ceil được sử dụng để làm tròn kết quả của phép chia thành 1 số nguyên int 
                    : int(np.ceil(float(w) / self.features_strides[idx])),  
                ] = 0 # các phần đệm này được gán = 0 = False để phân biệt với các giá trị không cần đệm 

            # Thêm danh sách các đệm feature_map này vào danh sách mask 
            masks.append(masks_per_feature_level)

        # cuối cùng là trả về danh sách chứa kết quả cần đệm cho các features map 
        return masks 
    

# sử dụng @META_architecture_Registry.register là một decorator dùng để 
# đăng ký lớp DETR vào một registry, thường được sử dụng trong các FrameWork học sâu để 
# quản lý các kiến trúc mô hình khác nhau 
@META_ARCH_REGISTRY.register()
# Thiết lập class DETR cần đăng ký 
class Detr(nn.Module):
    """Implement Detr"""
    # thiết lập phương thức khởi tạo và định nghĩa các tham số thuộc tính 
    def __init__(self, cfg):
        super().__init__()
        # Thiết lập thiết bị (CPU hoặc GPU) cho mô hình 
        self.device = torch.device(cfg.MODEL.DEVICE)
        # Số lượng lớp đối tượng mà mô hình sẽ nhận diện 
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        # có sử dụng mặt nạ phân vùng hay không 
        self.mask_on = cfg.MODEL.MASK_ON 
        # kích thước ẩn của mô hình Transformer 
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM 
        # số lượng các truy vấn đối tượng 
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Các tham số của transformer
        # 1: number header 
        nheads = cfg.MODEL.DETR.NHEADS
        # 2: dropout layer 
        dropout = cfg.MODEL.DETR.DROPOUT
        # 3: feed for ward dim 
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Các tham số cho việc tính toán mất mát
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        # l1
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        # 2 giám sát sâu 
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        # Tạo backbone và transformer cho mô hình
        N_steps = hidden_dim // 2 # số lượng steps của mô hình
        d2_backbone = MaskedBackbone(cfg) # Masked backbone CNN 
        # Lớp kết học backbone 
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        # lấy ra num_channels 
        backbone.num_channels = d2_backbone.num_channels
        # CẤU HÌNH CHO LỚP TRANSFORMR 
        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,

            # cấu hình cho ffn transformer 
            dim_feedforward=dim_feedforward,
                num_encoder_layers=enc_layers,
                num_decoder_layers=dec_layers,
                normalize_before=pre_norm,
                return_intermediate_dec=deep_supervision,
        )

        # Cấu hình DETR model 
        self.detr = DETR(
            backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, aux_loss=deep_supervision
        )
        # kiểm tra xem masks_on tức mặt nạ phân vùng có được thiết lập 
        if self.mask_on: 
            # nếu có ta đóng băng trọng số của DETR 
            frozen_weights = cfg.MODEL.DETR.FRONZE_WEIGHTS 
            # nếu như trọng số fronze != rỗng 
            if frozen_weights != '':
                # in ra thông tin cho biết cần tải trọng số đã được đào tạo trước 
                print('LOAD pre-trained weights') 
                # gán các trọng số được tải cho weight 
                weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
                # khởi tạo một từ điển weight để lưu trữ các weight sau khi được sử lý
                new_weight = {}
                # duyệt qua danh sách từ diển weight ban đầu 
                for k , v in weight.items():
                    # kiểm tra xem keys = detr có nằm trong từ điển 
                    if 'detr' in k : 
                        # gán lại cho keys có tên là detr = '' với các values được giữ nguyên 
                        new_weight[k.replace('detr','')] = v 
                    # trường hợp còn lại 
                    else: 
                        # In ra mà hình mọt thông tin cho biết bot qua  trọng số của mô hình đóng băng 
                        print(f"Skipping loading weight {k} from frozen model")
                # xóa từ điển weight khỏi quá trính 
                del weight
                # tải lại từ điển weight của mô hình detr cập nhật nó = new_weight
                self.detr.load_state_dict(new_weight)
                # sau đó ta lại thực hiện xóa bỏ từ điển new_weight cũ đi 
                del new_weight
            # thưucj hiện phân đoạn đối tượng
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
            # sau đó gọi hện PostProcessSegm để thực hiện tính toán kết quả phân đoạn cuối cũng qua một ngưỡng 
            # thresold để nhị phân hóa mặt nạ và thực hiện nội suy các mặt nạ lên kích thước của hình ảnh . 
            # SAU KHI MÔ HÌNH ĐÃ DỰ ĐOÁN CÁC FEATURES MAP CHO PHÂN ĐOẠN , CẦN PHẢI NỘI SUY 
            # CHÚNG CHỞ LẠI KÍCH THƯỚC BAN ĐẦU CỦA ẢNH , ĐỂ SO SÁNH CHÍNH XÁC VỚI GROUND TRUTH HOẶC 
            # HIỂN THỊ MẶT NẠ PHÂN ĐOẠN CHO HÌNH ẢNH GỐC. 
            self.seg_postprocess = PostProcessSegm

        # phân bổ detr cho thiết bị 
        self.detr.to(self.device)


        # xây dựng tiêu chuẩn đánh giá (criterion) 
        # sử dụng HungarianMatcher để ghép nối các dự đoán của mô hình với các GROUNTH TRUTH LABELS 
        #  Nó sử dụng thuật toán Hungarian (còn gọi là thuật toán Kuhn-Munkres) để tối ưu hóa việc ghép nối này dựa trên chi phí được định nghĩa.
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        # xây dựng 1 từ điển chứa các trọng số cho các loại loss khác nhau
        # gồm cross-entropy và loss_bounding box
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        # và loes G IOU 
        weight_dict["loss_giou"] = giou_weight
        # nếu như deep_supervision được kích hoạt 
        if deep_supervision:
            # tạo ra một từ điển trọng số phụ để áp dụng trọng số cho các lớp trung 
            # gian trong quá trình giám sát
            aux_weight_dict = {}
            # duyêt qua số lượng lớp decoder 
            for i in range(dec_layers - 1):
                # tạo ra các trọng số phụ cho mỗi lớp trung gian dựa trên số lượng lớp 
                # giảm mã 
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            # cập nhật trọng số phụ vào tưg điển weight_dict 
            weight_dict.update(aux_weight_dict)
        # à một danh sách các loại mất mát mà mô hình sẽ tính toán, bao gồm nhãn (labels), hộp giới hạn (boxes), và số lượng đối tượng (cardinality).
        losses = ["labels", "boxes", "cardinality"]

        # kiểm tra xem mặt nạ phân vùng có được thiếtlaapj 
        if self.masks_on: 
            # thêm vào từ điển loss 1 key = masks 
            # sử dụng hàm SetCriterion để tính toán các loss của DETR 
            self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
        )
        # phân bổ các phép tính chp GPU 
        self.criterion.to(self.device)

        # Lấy ra tham số pixels mean là 1 tensor được reshape = [3, 1, 1]
        # biểu diễn tb pixels trong hình ảnh 
        # các giá trị này là 1 tuple biểu diễn theo 3 kênh mày 
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        # và pixels std biểu diễn độ lệch chuẩn của accs pixels 
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # thực hiện chuẩn hóa các pixels trong hình ảnh 
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        # thực hiện phân bổ tính toán cho GPU 
        self.to(self.device)


    # Thiết lập phương thức forward phương thức này sẽ được gọi khi có các tham số được truyền vào 
    # tar về 1 từ điển là danh sách chứa các tensor 
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """

        # Thực hiện tiền xử lý dữ liệu hình ảnh cho mô hình DETR 
        images = self.preprocess_image(batched_inputs)
        # chuyển tiếp các hình ảnh cho DETR đã thiết lập 
        output = self.detr(images)

        if self.trainging: 
            # lấy accs thực thể  ground truth (gt_instances) từ dữ liệu đầu vapf (batched_inputs)
            gt_intences = [x['intences'].to(self.device) for x in batched_inputs]
            # chuẩn bị các mục tiêu target từ ground truth 
            targets = self.prepare_targets(gt_intences)
            # tính toán mất mát dựa trên dự đoán và mục tiêu 
            loss_dict = self.criterion(output, targets)
            # áp dụng trọng số (weight_dict) cho từng loạt mất mát trong loss_dict 
            weight_dict = self.criterion.weight_dict 
            # áp dụng trọng số cho mất mát 
            # duyệt qua danh sách các loss 
            for k in loss_dict.keys():
                # kiểm tra xem keys này có tồn tại trong từ điển  weight_dict hay không 
                if k in weight_dict: 
                    # nhân trọng số weight theo k với giá trị k 
                    loss_dict[k] *= weight_dict[k]
            # Trả về từ điển mất mát
            return loss_dict
        
        else: # trong quá trình suy luận 
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            # lấy dự đoán mặt nạ nếu có 
            mask_pred = output["pred_masks"] if self.masks_on else None 
            # thực hiện suy luận (self.ìnerence) để lấy kết quả của dự đoán 
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes)
            processed_results = []
            # xử lý kêt quả cho từng ảnh 
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_size):
                # lấy ra kichd thước của hình ảnh 
                height = input_per_image.get("height", image_size[0])
                # và kích thước w của ảnh 
                width = input_per_image.get("width", image_size[1])

                # áp dụng hậu xử lý (detector_postprocess) cho kết quả dự đoán của từng ảnh 
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            # trả về kết quả đã sử lý 
            return processed_results 
        

    # Thiết lập phương thức prepare_targets để chuẩn bị các mục tiêu target 
    # cho mô hình DETR 
    def prepare_targets(self, targets):
        # tạo một danh sách target mới = nONE 
        new_targets = None 

        # duyệt qua dnah sách accs mục tiêu từ từ điển đầu vào 
        for targets_per_image in targets: 
            # lấy ra kích thước h và w của image 
            h, w = targets_per_image.image_size     
            # tạo một tensor 4 chiều chứa kích thước ảnh để chuẩn hóa hộp giới hạn 
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            # lấy ra lớp nhẫn của ground truth 
            gt_classes = targets_per_image.gt_classes
            # chuẩn háo hộp giới hạn ground truth và chuyển đổi sang định dạng 
            # [cx, cy, w, h]
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            # chuyển đổi sang tạo độ cx,cy, w, h
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # thêm lables và boxes vào danh sách targets dưới dạng từ điển 
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            #  Nếu mô hình được cấu hình để sử dụng mặt nạ và mục tiêu có mặt nạ
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                # # Lấy mặt nạ ground truth và chuyển đổi từ định dạng polygon sang mặt nạ
                gt_masks = targets_per_image.gt_masks
                # chuyển đổi mặt nạ từ định dnagj polysgon thành masks 
                # định dạng polysgon là định dạng biểu diễn các đa giác chứa các tọa độ 
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                # cập nhật mục tiêu mới với mặt nạ 
                new_targets[-1].update({'masks': gt_masks})
        # trả về danh sách chứa các targets đã chuẩn bị 
        return new_targets
    
    # thiết lập phương thức suy luận 
    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # đảm bảo rằng độ dành của 2 danh sách sau như nhau
        assert len(box_cls) == len(image_sizes)
        # tạo một danh sách result để lưu trữ các kết quả 
        results = []
        # tính toán điểm số và nhãn cho mỗi hộp dự đoán 
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
        # xử lý từng hình ảnh trong batch 
        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            # thực hiện khởi tạo một đối tượng của lớp Instances 
            # đối tượng này sẽ chứa thông tin về accs dự đoán của mô hình như 
            # hộp giới han, nhãn lớp  điểm số, và có thể là mặt nạ, tất cả đều liên quan đến một ảnh cụ thể.
            result = Instances(image_size)
            # chuyển đổi hộp dự đoán từ cx, cy w, h sang xy,xy 
            # và điều chỉnh theo kích thước của hình ảnh  
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            # thực hiện châunr hóa các kích thước [0 -> 1]
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])

            # Nếu mô hình dự đoán mặt nạ
            if self.mask_on:
                # Chuyển đổi kích thước mặt nạ dự đoán và áp dụng ngưỡng để tạo mặt nạ nhị phân
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                # Cắt và điều chỉnh kích thước mặt nạ để phù hợp với hộp dự đoán
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                # thêm chiều mới vào vị trí thứ 2 của tensor có shape = masks_pred[0]
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            # Gán điểm số và nhãn lớp dự đoán cho kết quả
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results
    
    # Thiết lập phương thức xử lý dữ liệu 
    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        # thực hiện chuẩn hóa dữ liệu 
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        # sau đó chuyển đổi dnah sách anyf thành 1 tensor 
        images = ImageList.from_tensors(images)
        return images