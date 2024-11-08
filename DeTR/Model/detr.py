""" 
DETR model and criterion classes. 
Mô hình DETR và các lớp tiêu chí 

"""
import torch 
import torch.distributed
import torch.nn.functional as F 
from torch import nn 
from Utils import box_ops
from Utils.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy, 
                    get_world_size, interpolate, is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher 
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, 
                        dice_loss, sigmoid_focal_loss)

from .transformer import build_transformer

# Xây dựng lớp phương thức DETR là mô hình DETR 
class DETR(nn.Module):
    """This is the DETR module that performs object detection.
        Đây là module DETR thực hiện phát hiẹn đối tượng.
    """
    # Thiết lập phương thức định nghĩa các tham số thuộc tính lớp 
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model. 
        
        Parametres: 
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torxh module of the transformer architecture. See transformer.py 
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number 
                DETR can detect in a single image. For COCO, we recommend 100 queries. 
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        # định nghĩa các thuộc tính 
        self.num_queries = num_queries
        # transformer 
        self.transformer = transformer 
        # hidden_dim 
        hidden_dim = transformer.d_model 
        # Linear projection (embedding) 
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # lớp nhúng cho hộp giới hạn sử dụng kiến trúc sử lý mlp 
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # linear projection input Convolutional 2D
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    
    # Thiết lập phương thức forward phương thức này sẽ được gọi khi các tham số được truyền 
    # vào. Nhận đầu vào là samples một danh sách các nested Tensor 
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consits of:
            
                - samples.Tensor: batched images, of shape [batch_size *3 * h * w]
                - samples.mask: a binary mask of shape [batch_size * H * W], containing 1 on padded pixels 
        
            It returns a dict with the follwing elements: 
                - "pred_logits": the classification logits (including no-object) for all queries. 
                        shape = [batch_size * num_queries * (num_classes + 1)]
                        # Tức là đầu ra dự đoán thô của mô hình chứa số lượng các dối tượng queries 
                        # và đầu ra dự đoán của chúng 
                -"pred_boxes": The normalized boxes coordinates for all queries, represeted as
                        (center_x, center_y, height, width). These values are normalized in [0, 1],
                        relative to the size of each individual image (disregarding possible padding).
                        See PostProcess for information on how to retrieve the unnormalized bounding box.

                -"aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of 
                        dictionnaries containing the two above keys for each decoder layer.

            
        """
        # kiểm tra một trường hợp xem samples có phải là 1 list hay là 1 tensor 
        if isinstance(samples, (list, torch.Tensor)):
            # chuyển đổi samples thành dnah sách nestedTensor 
            samples = nested_tensor_from_tensor_list(samples)
        
        # sử dụng kiến trúc backbone cnn để lấy ra các đặc trưng và vị trí pos của các đặc trưng 
        features, pos = self.backbone(samples)
        # sử dụng decompose để phân dã tensor đặc trưng features cuối cùng
        # lấy ra thông tin src và mặt nạ đi kèm của nó 
        src, mask = features[-1].decompose()
        # đảm bảo rằng mặt nạ mask tồn tại 
        assert mask is not None
        # lấy result của transformer gán nó cho hs 
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # thục hiện nhúng thông tin hs 
        outputs_class = self.class_embed(hs)
        # dự đoán tọa độ cho hộp giới hạn các tọa độ này sẽ nằm trong khoảng 0 -> 1
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # thêm vào các thông tin vào từ điển bao gồm xác suất lớp và tọa độ bounding box 
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        #  Nếu mô hình được cấu hình để sử dụng auxiliary loss, thì thêm thông tin này vào đầu ra.
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
    

    # @torch.jit.unused: Đánh dấu hàm _set_aux_loss để không được sử dụng bởi TorchScript 
    # (được sử dụng khi chuyển đổi mô hình sang định dạng có thể triển khai).
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # đây là một cách giải quyết để làm cho torchscrip trở nên vui vẻ, vì 
        # torchscript hông hỗ trợ từ điển các giá trị không đồng nhất chẳng hạn như 
        # một lêngj có cả Tensor và list 
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    
    The process happens in two steps:
        1) we compute hungarian assigment between ground truth boxes and the outputs of the model 
        2) we supervise each pair of matched ground truth / prediction (superise class and box)
    
    """
    # Thiết lập phương thức khởi tạo nhận đầu vào gồm số lớp phân loại ,
    # matcher để tính toán sự phù hợp giữa mục tiêu với đề xuât , weight_dict từ điển 
    def __init__(self, num_classes,matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        
        Parameters: 
            num_classes: number of object categories, omitting the special no-object category 
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key names of the loss and as values their relative weight. 

            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        
        """ 
        # Định nghĩa các thuộc tính trên 
        super().__init__()
        # Số lượng lớp đối tượng , không bao gồm không có đối tượng 
        self.num_classes = num_classes
        # Mô-đun để tính toán sự phù hợp giữa các đối tượng và đề xuất 
        self.matcher = matcher 
        # từ điển chứa trọng số tương đối của các loại loss
        self.weight_dict = weight_dict
        # trọng số tương đối cho lớp không có đôí tượng 
        self.eos_coef = eos_coef
        # danh sách các loại loss sẽ được áp dugj
        self.losses = losses 
        # Tạo một tensor trọng số với tất cả các giá trị là 1 
        emty_weight = torch.ones(self.num_classes + 1)
        # Đặt trọng số cho lớp không có đối tượng = -1
        emty_weight[-1] = self.eos_coef

        # Đăng ký tensor trọng số emty_weight như 1  buffer
        # nhằm mục đích không muốn tính toán gradient 
        self.register_buffer("emty_weight", emty_weight)


    #Thiết lập phương thức deff los để tính toán tổn thất cho nhãn 
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
            targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # kiểm tra để đảm bảo rằng pred_logits có trong từ điển outputs
        assert 'pred_logits' in outputs
        # lấy logits dự đoán đầu ra 
        src_logits = outputs['pred_logits']

        # lấy chỉ số sau khi áp dụng phép hoán vị từ 'indices' 
        idx = self._get_src_permutation_idx(indices)
        # ghép các nhãn đích từ ground truth box đã được gán cho các dựu đoán 
        target_classes_0 = torch.concat([t["labels"][j] for t, (_,j) in zip(targets, indices)])
        # tạo một tensor chứa chỉ số lớp đích với giá trị mặc định là số lượng lớp 
        # ( đối với các trường hợp không có đối tượng)
        # lấp đầy 1 tensor với shape = 2 kích thước đầu của tensor src_logits với các 
        # phần tử của self.num_classes 
        target_classess = torch.full(src_logits.shape[:2], self.num_classes)
        # gán các nhãn thực tế vào tensor target_class tại các vị trí tương ứng 
        target_classess[idx] = target_classes_0 

        # Tính toán hàm lỗi cross_entropy giữa logits dự đoán và nhãn đích 
        # tensor chứa các dự đoán src_logits sẽ được chuyển vị 
        loss_ce = F.cross_entropy(input = src_logits.transpose(1,2), target=target_classess, weight=self.emty_weight)
        # thêm các giá trị này vào từ điển losses 
        losses ={'los_ce': loss_ce}

        # kiểm tra xem log có = True 
        if log: 
            # cần xây dựng một hàm lỗi riêng biệt cho phần này 
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_0[0])

        # Trả về từ điển chứa các hàm lỗi 
        return losses 
    

    # sử dụng @torch.no_grad để bỏ qua việc sử dụng gradient descent cho 
    # phương thức loss_cardinality 
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Tính toán lỗi số lượng , tức là lỗi tuyệt đối trong số lượng hộp dự đoán 
                không rỗng. 

        Đây không phảỉ là một hàm lỗi thực sự nó chỉ dùng cho mục đích ghi nhận. Nó không 
            lan truyền gradient
        
        """
        # lấy logits dự đoán từ đầu ra của mô hình 
        pred_logits = outputs['pred_logits']
        # lấy ra thiết bị mà logits đang sử dụng 
        device = pred_logits.device
        # tính độ dài mục tiêu, tức là số lượng nhãn  thức tế cho mỗi hình ảnh 
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        # đếm số lượng dự đoán không phải là "không có đối tượng" (lớp cuối cùng)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] -1).sum(1)
        # tính lỗi L1 giữa số lượng dự đoán và số lượng thực tế 
        card_err =F.l1_loss(card_pred.float(), tgt_lengths.float())
        # lưu vào từ điển 
        losses = {"cardinality_error": card_err}

        return losses
    

    # Tính toán lỗi liên quan đến các hộ giới hạn 
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
            Tính toán các hàm lỗi liên quan đến hộp giới hạn bounding boxes, hàm lỗi L1 
                regression và hàm lỗi GIoU. 
            Các từ điển mục tiêu phải chứa khóa 'boxes' là 1 tensor có shape [number_target_boxes, 4] các hộp 
                mục tiêu được kỳ vọng với định dạng [cx, cy , w, h], chuẩn hóa theo kích thước hình ảnh 
        """
        """
            Indices chứa các thông tin quan trọng được sử dungh để theo dõi sự tương ứng giữa các dự đoán của mô hình 
                 và các ground truth boxes trong dữ liệu mục tiêu. Cụ thể, indices chứa thông tin về cách ghép các 
                 dự đoán với các ground truth boxes dựa trên phép gán Hungarian. 

            indices thường là một tuple chứa hai tensor. Mỗi tensor trong tuple này biểu diễn một phần của phép gán:
                Tensor thứ nhất chứa chỉ số của các dự đoán.
                Tensor thứ hai chứa chỉ số của các ground truth boxes tương ứng.    
        """
        # đảm bảo giằng pred_boxes trong từ điển outputs 
        assert 'pred_boxes' in outputs 
        # lấy ra các chỉ sô sau khi áp dụng phép hoán vị từ indices 
        idx = self._get_src_permutetation_idx(indices)
        # ánh xạ các giá trị pred_boxes trong từ điển output theo chỉ số y 
        # cho danh sách src_boxes [danh sách này sẽ chứa các nhãn dự đoán cho hộp giới hạn ]
        src_boxes = outputs['pred_boxes'][idx]
        # lọc ra danh sách các target của boxes từ từ điển và chỉ số indices tương ứng 
        # thêm lần lượt vào target_boxes 
        target_boxes = torch.cat([t['boxes'][i] for t , (_,i) in zip(targets, indices)], dim=0)

        # tính toán tổn thất l1 cho các hộp giới hạn 
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reducation='none')

        # xây dựng 1 từ điển loss = None để lưu trữ accs giá trị loss được tính toán
        losses = {}
        # tính toán tổng tổn thất l1 trung bình cho các hộp giới hạn 
        # và thêm nó vào từ điển 
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # Tính toán tổn thấy IOu 
        # 1 sử dụng hàm torch.diag để lấy đường chéo ma trận, tức là, nó lấy 
        # các giá trị GIoU cho mỗi cặp bounding box dự đoán và ground truth tương ứng 
        # 2: box_ops.generalized_box_iou. Tính toán GIoU giữa 2 tập hợp bounding_box. 
        # GIoU mở rộng Iou bằng cách tính cả trường hợp hai hộp không giao nhau 
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        # cuối cùng tính toán trung bình tổn thất và thêm nó vào từ điển losses
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        # trả về kết quả nằm trong từ điển losses 
        return losses
    

    # Xây dựng phương thức tính chi phí cho masks 
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """"
        Compute the losses related to the masks: the focal loss and the dice loss.
            targets dicts must contain the key "mask" containing a tensor of dim [nn_target_boxes,h ,w]

        """
        # đảm bảo pred_masks có tồn tại trong từ điển outputs 
        assert "pred_masks" in outputs

        # lấy ra các chỉ sổ sau khi áp dụng phép hoán vị cho indices 
        # 1: chỉ số cho src_idx 
        src_idx = self._get_src_permutation_idx(indices)
        # 2: chỉ số của tgt_idx 
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # ấy ra values pred_masks từ từ điển outputs và gán nó cho src_mask 
        src_masks = outputs["pred_masks"]
        # gán các giá trị theo chỉ số src_idx của danh sách src_masks(chứa các masks được dự đoán)
        src_masks = outputs['pred_masks']
        # duyệt qua từ điển target lấy ra các masks lưu chúng vào danh sách biến = masks
        masks =[t["mask"] for t in targets]

        # chuyển đổi danh sách các tensor mặt nạ thành một NestedTensor, sau đó phân dã nó thành 
        # target_masks chứa các mặt nạ mục tiêu, và valid là một tensor chỉ ra những vùng hợp lệ 
        # không bị ảnh hưởng bởi padding. 
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # Chuyển đổi target_masks để có cùng kiểu dữ liệu và thiết bị(ví dụ: CPU hoặc GPU)
        # với src_masks, là tensor chứa các mặt nạ dự đoán của mô hình. 
        target_masks = target_masks.to(src_masks)
        # lọc target_masks sử dụng chỉ số tgt_idx, có thể là chỉ số của những cùng mục tiêu cần được 
        # so sánh với dự đoán 
        target_masks = target_masks[tgt_idx]
        # Nội suy src_masks để có kích thước bằng với target_masks sử dụng phương pháp 
        # nội suy bilinear 
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        # loại bỏ chiều thứ 2 (được thêm vào bởi none) và làm phẳng tensor để chuẩn bị cho việc 
        # tính toán mất mát 
        src_masks = src_masks[:, 0].flatten(1)
        # thay đổi hình dnagj của tensor target_mask bằng với 
        # kích thước src_masks.shape 
        target_masks = target_masks.view(src_masks.shape)
        # Tính toán tônt thất cho masks mất mát focal sử dụng hàm sigmoid (đặc biệt hữu ích khi có sự mất cân đối giữa các lớp)
        # và mất mất Dice(đo lường sự tương đồng giữa 2 mẫu)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        # cuối cùng trả về từ điển losses chứa các chi phí đã được tính toán 
        return losses 
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices 
        # indices là một tuple mỗi tuple chứa 2 phần tử : src và một phần từ khác không được sử dụng 
        # (được biểu diễn bằng _). src chứa các chỉ số của các dự đoán cần được hóan vị 

        # 1: Tạo một tensor batch_idx bằng cách nối các tensor. MỖi tensor trong danh sách được 
        # tạo ra bằng hàm torch.full_like, sao cho mỗi tensor có cùng kích thước với src và chứa giá trị i chỉ số 
        # của batch
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        # Trả về hai tensor batch_idx và src_idx. batch_idx chứa các chỉ số của batch,
        # và src_idx chứa các chỉ số của các dự đoán cần được hoán vị.
        return batch_idx, src_idx
    

    def _get_tgt_permutation_idx(self, indices):
        # permute  targets follwing indices 
        # tạo ra các chỉ số hoán vị cho các mục tiêu dựa trên indices, batch_idx
        # là một tensor chứa các chỉ số của batch , tgt_idx la tensor chứa các chỉ số mục tiêu. 
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])                                                          
        tgt_idx = torch.cat([src for (src, _) in indices])
        # returm batch_idx, src_idx
        return batch_idx, tgt_idx

    
    # Thiết lập phương thức get_loss 
    # để có thể hợp nhất các giá trị loss vào trong 1 từ điển duy nhất 
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # Xây dựng 1 từ điển loss map , lưu trữ tất cả các loss 
        # vào từ điển với khóa và giá trị tương ứng với nó 
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality, 
            'boxes': self.loss_boxes, 
            'masks': self.loss_masks 
        }
        # kiểm tra xem key loss có trong từ điển loss map 
        # nếu không đưa ra môtk cảnh báo 
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        # trả về từ điển loss_map với key loss = tất cả các keys trong từ điển 
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    

    # Thiết lapaj phuonwg thức forward để tính toán các loss biểu diễn 
    # phương thức này sẽ được gọi khi có các tham số được chuyền vào 
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        
        """
        # xây dựng 1 từ điển outputs_with_aux một từ điển không phụ trơh 
        # với các key và values là các tensor đầu ra trong từ điển outputs 
        outputs_without_aux = {k: v for k , v in outputs.items() if k != 'aux_outputs'}

        # truy suất sự trùng khớp giữa đầu ra lớp cuối cùng và và target
        indices = self.matcher(outputs_without_aux, targets)

        #tính toán trung bình của số lượng hộp giới hạn thông qua tất cả các nốt, cho 
        # mục đích bình thường hóa 
        # 1: tính toán số lượng hộp giới hạn trong từ điển target 
        num_boxes = sum(len(t['labels']) for t in targets)
        #  chuyển đổi num_boxes thành 1 tensor và 
        # phân bổ tensor này cho mỗi các thiết bị được sử của các tensor trong từ điển outputs 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # kiểm tra xem 1 giá trị khởi tạo có = True 
        if is_dist_avail_and_initialized: 
            # tính toán trung bình các hộp giới hạn 
            torch.distributed.all_reduce(num_boxes) 
        
        # sử dụng hàm clamp để tính toán các boxes trong danh sách và giới hạn 
        # các boxes với inputs = num_boxes / get_world_size [là số lượng thiết bị tham gia môi trường phân tán]
        # và min = 1 
        num_boxes = torch.clamp(num_boxes /get_world_size(), min=1 ).item()


        # Tính toán tất cả các tổn thất được yêu cầu '
        # 1: khởi tạo một từ điển los để lưu trữ tất acr các kết quả 
        losses= {}
        # duyệt qua các los trong danh sahc losses 
        for loss in self.losses: 
            # cập nhật các loss vào từ iển losses 
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))    
        # trong trường hợp tổn thất phụ, lặp lại các quá trình này với đầu ra của mỗi lớp trung gian 
        # kiểm tra xem aux_outputs [đầu ra phụ trợ] có nằm trong từ điển outputs hay không 
        if 'aux_outputs' in outputs: 
            # duyệt qua danh sách các lớp phụ trợ lấy ra các giá trị và chỉ số tương ứng của nó 
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # so sánh sự phù hợp của cac giá trị aux_outputs với từ điển target
                indices = self.matcher(aux_outputs, targets)
                # duyệt qua từ điển loss và lấy ra ácc loss 
                for loss in self.losses: 
                    # kiểm tra xem có phải là key masks 
                    if loss == 'masks':
                        # bỏ qua tổn thất mặt nạ trung gian vì quá trình tính toán quá tốn kém 
                        continue 
                    
                    # tạo một từ điển để lưu trữ logging chỉ co thể cho lớp cuối cùng 
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    # khởi tạo 1 danh sách lưu trữ các loss
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    # thêm các giá trị từ danh sách và chỉ số của nó vào từ điển 
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    # cuối cùng là cập nhật từ điển losses với values là từ điển l_dict 
                    losses.update(l_dict)
        # trả về từ điển losses
        return losses
    


# Thiết lập lớp phương thức PostProcess để thực hiện xử lý sau trình tự 
# chuyển đổi cac đầu ra của mô hình thành định dạng được mong đợi bởi coco api 
class PostProcess(nn.Module):
    """This modul converts the model's output into the format expected by the coco api"""
    # bỏ qua việc tính toán gradient descent cho lớp này 
    @torch.no_grad()
    # với phương thức forward sẽ biểu diễn các tính toán phương thức này sẽ được gọi 
    # khi các tham số được truyền vào 
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # lấy ra các nhật ký dự đoán và hộp dự đoán 
        # pred_logits là lớp nhật ký phân loại ghi lại tất cả các thông tin phân loại 
        # của các truy vấn shape =[batch_size , num_queries, classes + 1]
        # tức là đầu ra thô của mô hình trước khi tính toán soft_max
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # đảm bảo rằng dộ dài của danh sách ghi nhật ký bằng độ dài danh sachd target_size 
        assert len(out_logits ) == len(target_sizes)
        # và target_size dim(1) = 2

        # xây dựng một hàm softmaxt  tính toán các giá trị của out_logits, - 1
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        # sử dụng hàm unbind để tách tensor target_size theo chiều thứ 2
        img_h, img_w = target_sizes.unbind(1)
        # sau đó nối các giá trị này thành 1 dánhachs 
        # SHAPE = [batch_Size , 4]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # nhân boxes với tensor scae factor để chuyển các tọa độ tương đối [0,1] sang 
        # tọa độ tuyệt đối 
        # bằng cách mở rộng tensor scale_fct shape = [batch_size, 1 , 4]
        # nhân với tensor boxes shape = [batch_size, num_boxes, 4]
        boxes = boxes * scale_fct[:, None, :]

        # lưu tất cả các kết quả vào từ điển results
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


# Xây dựng lớp MLP lớp đa xử lý tương tự như các khối ffn layerb
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    # Thiết lập phuwong thức khưởi tạo và đingh nghĩa các thuộc tính 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

# buil model DETR 
def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # gán cho num_classses = 20 nếu data_dile không phải là coco 
    num_classes = 20 if args.dataset_file != 'coco' else 91
    # nếu data)file trong từ điển tham số là cocopanoptic 
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        # gán số lớp = 250 
        num_classes = 250
    # lấy ra divice từ từ điển tham sóo của mô hình 
    device = torch.device(args.device)
    # xây dựng mô hìnhback bone 
    backbone = build_backbone(args)

    # xây dựng mô hình Transformer 
    transformer = build_transformer(args)

    # sau đó xây dựng mô hình DETR 
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    # kiểm tra xem trong từ điển mô hình có tồn tại masks
    if args.masks:
        # nếu có maskd thực hiện detr masked model segmentation 
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    # build_matcher model 
    matcher = build_matcher(args)
    # lấy ra các trọng số của mô hình gồm loss và bounding box loss 
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    # sau đó là iou loss cho và gán tất cả vào từ điển weight_dict 
    weight_dict['loss_giou'] = args.giou_loss_coef
    # 
    if args.masks:
        # lấy ra loss của mask tương tự như các loss trên thêm vào từ điển wieght_dict 
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    # với chi phí phụ 
    if args.aux_loss:
        # 1 từ điển để lưu trữ các wieght phụ 
        aux_weight_dict = {}
        # duyệt qua các dect layers 
        for i in range(args.dec_layers - 1):
            # cập nhật từ điển aux với các gia trị weight_dict 
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        # 
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    # SỬ LÝ CÁC MẶT NẠ SAU KHI TÍNH LOSS 
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        # PHÂN ĐOẠN ĐỐI TƯỢNG 
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            # LẤY RA CÁC IS_THING_MAP là một từ điển các ID lớp và giá trị là một boolean chỉ 
            # ra lớp đó là thing(True) hay stuff(False)
            is_thing_map = {i: i <= 90 for i in range(201)}
            # CHUYỀN VÀO BƯỚC SỬ LÝ PANOPTIC ĐỂ TRẢ VỀ CÁC DỰ ĐOÁN CUỐI CÙNG 
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
