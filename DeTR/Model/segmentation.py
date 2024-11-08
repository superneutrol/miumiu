import os 
import io 
from collections import defaultdict 
from typing import List, Optional

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor 
from PIL import Image           

import Utils.box_ops as box_ops 
from Utils.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try: 
    from panopticapi.utils import id2rgb , rgb2id  

except ImportError: 
    pass 


# Thiết lập lớp phương thức Segmentation DETR thực hiện nhiệm vụ phân đoạn đối 
# tượng 

class DETRsegm(nn.Module):
    # Thiết lập phương thức khởi tạo  và định nghĩa các thuộc tính lớp 
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        # định nghĩa một thuộc tính DETR là mô hình DETR 
        self.detr = detr 

        # Nếu như freeze_detr = True 
        if freeze_detr: 
            # duyệt qua danh sách các tham số trong từ điển parameters 
            for p in self.parameters():
                # bỏ qua việc yêu cầu gradient cho các tham số 
                p.requires_grad(False)

        #lấy ra kích thước nhúng và number_head của DETR 
        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        # Multihead-Attention-Map để trích xuất các feature map của hình ảnh theo các tao độ 
        # và tính toán attention cho mỗi feature map 
        self.bbox_attention = MHAttentionMap()
        # Masked Head smallConv một lớp fully connected để thực hiện masked các đặc trưng 
        # feature map 
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)


    # Thiết lập phương thức xử lý forward , phuuwong thức này sẽ được gọi khi 
    # các tham số được truyền vào 
    def forward(self, samples: NestedTensor):
        # kiểm tra xem tensor samples có phải là 1 trong 2 trường hợp biểu diễn 
        # là list hoặc Tensor 
        if isinstance(samples, (list, torch.Tensor)):
            # nếu thỏa mãn áp dụng phép biến đổi NestedTensor từ danh sách các tensor 
            samples = nested_tensor_from_tensor_list(samples)
        
        # áp dụng DETR.backbone để lấy ra các đặc trưng và vị trí của nó trong ảnh 
        # các features ở đây có thể hiểu là [features map]
        features, pos = self.detr.backbone(samples)

        # lấy ra kích thước của tensor features cuối cùng theo chiều thứ nhất
        # gán nó cho bs [vì tensor cuối cùng có độ phân giảii cao hơn và có lượng thông tin được tích hợp]
        bs = features[-1].tensor.shape[0]

        # phân dã đặc trưng cuối cùng thành src [nguồn] và mask [mặt nạ]
        # sử dụng decompose để phân dã nó 
        src , mask = features[-1].decompose()
        # đảm bảo giằng mask có tồn tại 
        assert mask is not None
        
        # chuẩn bị đầu vào cho transformer bằng 1 phéo chiếu tuyến tính lên 
        # không gian biểu diễn tensor src 
        src_proj = self.detr.input_proj(src)
        # áp dụng Transformer cho src tuyến tính lấy ra Result và Weight [tham số biểu diễn]
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1]) # tái sử dụng lại pos của features[-1]

        # Thực hiện việc dự đoán các tọa độ 
        # 1: Nhúng sâu đầu ra biểu diễn Result của transformer để dự đoán cho lớp của đối tượng
        outputs_class = self.detr.class_embed(hs)
        # 2: Từ biển diễn nhúng sâu Result áp dụng tính toán tọa độ hộp giới hạn , với 
        # hàm sigmoid đưa các tạo độ biểu diễn này trong [0->1]
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        
        # gán cho từ điển out với keys = pred_logits và pred_box [lớp phân loại và hộp giới hạn của đối tượng]
        # và values là nhãn lớp của đối tượng cuối cùng và hộp giới hạn của nó 
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        # Nếu self.detr.aux_loss là True, đầu ra phụ sẽ được thêm vào từ hàm _set_aux_loss.
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)


        # Tính toán chú ý attention cho features hs[-1][hidden_dim cuối] và tái sử dụng memory [weight]
        # của transformer . để tính toán chú ý cho hộp giới hạn và tạo ra mặt nạ ch bouding bõ 
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        # Tạo mặt nạ phân đoạn: self.mask_head nhận đầu vào là src_proj, 
        # bbox_mask, và các tensor đặc trưng để tạo ra mặt nạ phân đoạn.
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])

        # Định hình lại và trả về kết quả: Mặt nạ phân đoạn được định hình lại và thêm vào đầu ra dưới dạng pred_masks.
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        out["pred_masks"] = outputs_seg_masks
        return out


# Thiết lập phương thức expand để thêm 1 chiều mới có kích thước 
# được chỉ định length vào tensor nguồn 
def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach

    Đầu conVolutional đơn giảm , sử dụng nhóm norm, 
    Việc lấy mẫu thực hiện bằng cách sử dụng phương pháp FPN . 

    [FPN là phương pháp được sử dụng để tạo ra các bản đồ tính năm feature map FPN sử dụng 1 con đường 
        đi từ dưới lên để tính toán các bản đồ tính năng của backbone và 1 con đường đi từ trên xuống để kết hợp các 
        bản đồ tính năng có độ phân giải thấp và cao thông qua kết nối bên. FPn cho phép phát hiện và phân vùng các đối tượng
        ở nhiều kích thước khác nhau một cách hiệu quả.]
    """
    # thiết lập phuuwong thức khởi tạo và định nghĩa các thuộc tính tham số 
    def __init__(self, dim , fpn_dims, context_dim):
        super().__init__()

        # định nghĩa một thuộc tính inter dim là một danh sách chứa các dim giảm dần theo
        # dủa dim 
        inter_dims = [dim , dim // 2 , dim // 4 , dim // 8, dim // 16, dim // 64]
        # Đingh nghĩa các lớp thuộc tính Convolutional 2D 
        self.lay1 = torch.nn.Conv2d(dim, dim , 3, padding=1)
        # 2: Một lớp GroupNorm 
        self.gn1 = torch.nn.GroupNorm(8, dim)
        # 3: +1 layer Conv2d
        self.lay2 = torch.nn.Conv2d(inter_dims[1], 3, padding=1)
        # 4: +1 GroupNorma layer
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        # 5: +1 layer Conv2d
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        # 6: +1 GroupNorma layer 
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        # 7: +1 Layer Conv2d
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        # 8: +1 GroupNorma layer
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        # 9: +1 Layer Conv2d 
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        # 10: +1GroupNorma layer 
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        # 11: Final Layer conv2d
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)


        # Định nghĩa thuộc tính dim 
        self.dim = dim


        # Định nghĩa 3 thuộc tính adaption thực hiện biến chuyển đổi các lớp 
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        # duyệt qua 1 danh sách các module 
        for m in self.modules():
            # kiểm tra tra xem modules m có phải là layer Conv2d hay không 
            if isinstance(m , nn.Conv2d):
                # nếu m được đảm bảo là module Conv2d 
                # áp dụng khởi tạo kaiming_uniform. Phương pháp này giúp tránh vấn đề việc gradient biến mất hoặc bùng nôt 
                # trong quá trình huấn luyện mạng neural. Tham số a=1 được ử dụng cho hàm kích hoạt relu 
                nn.init.kaiming_uniform_(m.weight, a=1)
                # Và khởi tạo bias cho trọng số của lớp Conv2D với giá trị hằng số là 
                # Điều này đảm bảo rằng ban đầu bias không ảnh hưởng đến đầu ra của Convolutional
                nn.init.constant_(m.bias, 0)



    # Thiết lập phương thức forward để thực hiện xây dựng xử lý lớp 
    # phương thức này sẽ được gọi ngay khi có tham số truyền vào nó 
    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor] ):
        # xử lý tensor đầu vào x là danh sách chứa các features map và mặt nạ mask
        # với việc nối tensor x sẽ theo kích thước bbox_mask.flatten(0,1)
        x = torch.cat([_expand(x, bbox_mask.shape[1]),    bbox_mask.flatten(0,1)], 1)
        
        # áp dụng các lớp layer Conv2 và sử dụng các lớp chuẩn hóa GroupNorm để nhóm
        # các vùng ảnh trong một feature map riêng lẻ và tính toán trên các vùng ảnh đó 
        x = self.lay1(x)
        x = self.gn1(x)
        # Sau mỗi 2 lớp layer Conv2d and GroupNorma áp dụng 1 lớp kích hoạt relu 
        x = F.relu(x)
        # thực hiện đưa x qua 1 lớp mạng Conv2d và một lớp GropNorma như trên 
        x = self.lay2(x)
        x = self.gn2(x)
        # Và hàm kích hoạt  của nó 
        x = F.relu(x)


        # Thực hiện các bước xử lý trên các feature map được tạo ra bởi FPN 
        # 1 : Lấy ra features map đầu tiên từ FPN và áp dụng 1 adapter (Conv2d) để biến 
        # đổi thành một dnagj thích hợp cho việc xử lý tiếp theo . 
        cur_fpn = self.adapter1(fpns[0])
        # 2 : kiểm tra xem kich thước của cur_fpn có khác với kích thước của x hay không '
        if cur_fpn.size[0] != x.size(0):
            # Nếu không thì ta phải thực hiện điều chỉnh kích thước của x
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))

        # 3: cộng feature hiện tại với X đã được nỗi suy lên kích thước FPN 
        # để đảm bảo x và cur_FPn có cùng kích thức
        x = cur_fpn + F.interpolate(x , size=cur_fpn.shape[-2:], mode="nearest")
        # 4: áp dụng các lớp xử lý tiếp theo bao gồm cả chuẩn hóa 
        x = self.lay3(x)
        x = self.gn3(x)
        # 5: Áp dụng hàm kích hoạt relu để thêm tính phi tuyến tính và giúp mô hình 
        # học được các biểu diễn phức tạp 
        x = F.relu(x)


        # 1 : Lấy ra features map đầu tiên từ FPN và áp dụng 1 adapter (Conv2d) để biến 
        # đổi thành một dnagj thích hợp cho việc xử lý tiếp theo . 
        cur_fpn = self.adapter2(fpns[1])
        # 2 : kiểm tra xem kich thước của cur_fpn có khác với kích thước của x hay không '
        if cur_fpn.size[0] != x.size(0):
            # Nếu không thì ta phải thực hiện điều chỉnh kích thước của x
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))

        # 3: cộng feature hiện tại với X đã được nỗi suy lên kích thước FPN 
        # để đảm bảo x và cur_FPn có cùng kích thức
        x = cur_fpn + F.interpolate(x , size=cur_fpn.shape[-2:], mode="nearest")
        # 4: áp dụng các lớp xử lý tiếp theo bao gồm cả chuẩn hóa 
        x = self.lay4(x)
        x = self.gn4(x)
        # 5: Áp dụng hàm kích hoạt relu để thêm tính phi tuyến tính và giúp mô hình 
        # học được các biểu diễn phức tạp 
        x = F.relu(x)
        

        # 1 : Lấy ra features map đầu tiên từ FPN và áp dụng 1 adapter (Conv2d) để biến 
        # đổi thành một dnagj thích hợp cho việc xử lý tiếp theo . 
        cur_fpn = self.adapter3(fpns[2])
        # 2 : kiểm tra xem kich thước của cur_fpn có khác với kích thước của x hay không '
        if cur_fpn.size[0] != x.size(0):
            # Nếu không thì ta phải thực hiện điều chỉnh kích thước của x
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))

        # 3: cộng feature hiện tại với X đã được nỗi suy lên kích thước FPN 
        # để đảm bảo x và cur_FPn có cùng kích thức
        x = cur_fpn + F.interpolate(x , size=cur_fpn.shape[-2:], mode="nearest")
        # 4: áp dụng các lớp xử lý tiếp theo bao gồm cả chuẩn hóa 
        x = self.lay5(x)
        x = self.gn5(x)
        # 5: Áp dụng hàm kích hoạt relu để thêm tính phi tuyến tính và giúp mô hình 
        # học được các biểu diễn phức tạp 
        x = F.relu(x)


        # Cuối cùng là áp ápdungj 1 lớp Tích chập 2D cuối cùng cho x và trả về x 
        x = self.out_lay(x)
        return x 
    
        


    
# Xây dụng lớp phương thức Multi-Head - Attention Map 
# để tính toán Attention chp các features map 
class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""
    # Thiết lập phương thức khởi tạo và định nghĩa các tham số thuộc tính 
    def __init__(self, query_dim, num_heads, hidden_dim, dropout=0.0, bias=True):
        super().__init__()
        # định nghĩa các thuộc tính '
        # 1: num_heads 
        self.num_heads = num_heads
        # 2: hidden_dim 
        self.hidden_dim = hidden_dim
        # 3: Dropout 
        self.dropout = nn.Dropout(dropout)

        # 4: Queries Vector 
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        # 5: Key Vector 
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        # Khởi tạo tham số cho q , k vector 
        # 1: Khởi tạo danh sách bias [ q , k = zeros tensor ] 
        nn.init.zeros_(self.k_linear)
        nn.init.zeros_(self.q_linear.bias)
        # 2: Trọng số weight cho q, k sử dụng phân phối đồng nhất xavier giúp mô hình 
        # cải thiện tốc độ học 
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)   

        # và xây dựng một hệ số chuẩn hóa normalize fact 
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    
    # Xây dựng phương thức forward , phương thức này sẽ được gọi khi 
    # có các tham số truyền vào . 
    # có chức năng thực hiện tính Attention cho các features map 
    def forward(self, q, k, mask: Optional[Tensor] = None):
        # thực hiện tính toán vector q 
        q = self.q_linear(q)
        # áp dụng một lớp Conv2d LÊN VECTOR K 
        k = F.conv2d(k, weight=self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), bias = self.k_linear.bias )
        #  Định hình lại tensor q để phân chia các đầu vào thành self.num_heads nhóm đầu vào, shape  q = [batch_size,  num_features, num_head , head_dim]
        # mỗi nhóm có kích thước là self.hidden_dim // self.num_heads.
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        # Tương tự như qh, tensor k cũng được định hình lại để phù hợp với cơ chế multi-head attention. 
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        # Sử dụng phép tính Einstein summation để tính toán điểm số attention giữa qh và kh, 
        # sau khi đã nhân qh với một hệ số chuẩn hóa self.normalize_fact.
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            #  Nếu có một mask được cung cấp, điểm số attention sẽ được điều chỉnh để loại bỏ những 
            # vị trí không mong muốn bằng cách gán giá trị -inf vào những vị trí tương ứng trong mask.
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        # Áp dụng hàm softmax để chuẩn hóa điểm số attention, chuyển chúng thành các xác suất.
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        # Áp dụng dropout để giảm overfitting trong quá trình huấn luyện.
        weights = self.dropout(weights)
        return weights



# Thiết lập phương thywcs dice_loss thực hiện 
# tính toán dice los tương tự như IOU tổng quát cho masks 
def dice_loss(inputs, targets, num_boxes):
    """"
    compute the DICE loss, similar generalized IOU for mask
    Args: 
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
            Một tensor float có hình dạng tùy ý. Dự đoán cho từng ví dụ mục tiêu 

        
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

            Một float tensor có kích thước cùng với đầu vào. Lưu trữ nhãn phân loại nhị phân 
            cho từng phần tử trong đầu vào (0  cho lớp tích cực và 1 cho lớp tiêu cực)

    
    """
    # áp dụng phép biến đổi sigmoid (hàm nhị phân) lên danh sachs tensor đầu vào 
    inputs = inputs.sigmoid()
    # làm phẳng tensor input từ chiều thứ 2
    inputs = inputs.flatten(1)
    # Tính toán tích 2 lần tensor inputs và target lên chiều thứ 2
    # điều này đo lường sự chồng chéo giữa dự đoán và mục tiêu thực tế 
    numerator = 2 *(inputs * targets).sum(1)
    # Sau đó cộng tổng của 2 tensor này theo chiều cuối cùng  tensor sẽ có hình dạng mới 
    # nếu inputs shape = [b , n , h] trả về shape = [b, n] mỗi phần tử sẽ là tổng theo chiều chỉ định
    denominator = inputs.sum(-1) + targets.sum(-1)
    # Tính toán tổn thất
    #  Tính DICE loss cho mỗi ví dụ. Cộng thêm 1 vào tử số và mẫu số để tránh chia cho 0 và làm mượt phép tính. 
    loss = 1 - (numerator + 1) / (denominator + 1)
    #  Tính trung bình mất mát DICE trên tất cả các hộp (boxes) hoặc đối tượng trong batch.
    return loss.sum() / num_boxes





# Thiết lập phương thức sigmoid_focal_loss 
# để giải quyết vấn đề mất cân bằng giữa các lớp (class imbalance) khi có nhiều ví dụ tiêu cực (background) hơn so với ví dụ tích cực (object). 

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    #  Áp dụng hàm sigmoid lên inputs để chuyển đổi giá trị logits thành xác suất, nằm trong khoảng từ 0 đến 1.
    prob = inputs.sigmoid()
    # Tính cross-entropy loss giữa inputs và targets mà không áp dụng giảm tổng (reduction), 
    # tức là giữ nguyên kích thước ban đầu của tensor.
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # Tính xác suất của dự đoán đúng, p_t, cho mỗi phần tử. Nếu mục tiêu là 1, 
    # p_t sẽ là prob; nếu mục tiêu là 0, p_t sẽ là 1 - prob
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # : Áp dụng modulating factor (1 - p_t) ** gamma lên cross-entropy loss để tăng cường mất mát 
    # cho những ví dụ khó (hard examples) mà mô hình dự đoán sai.
    loss = ce_loss * ((1 - p_t) ** gamma)

    #  Kiểm tra xem có áp dụng trọng số alpha hay không
    if alpha >= 0:
        #  Nếu có, tính toán trọng số alpha_t cho mỗi phần tử dựa trên mục tiêu, và áp dụng nó lên mất mát.
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        # 
        loss = alpha_t * loss
    #
    # Tính trung bình mất mát trên mỗi ví dụ, tổng hợp lại, và chia cho tổng số hộp (boxes) để chuẩn hóa mất mát trên mỗi hộp.
    return loss.mean(1).sum() / num_boxes


# Xây dựng lớp thức PostProcessSegm được sử dụng để xử lý sau khi dự đoán các phân đoạn của 
# mô hình DETR 
# Lớp PostProcessing kế thừa nn.Module của Pytorch và khởi tạo một ngưỡng Threshold 
# mặc định = 0.5. Ngưỡng này sẻ sử dụng để quyết định một pixel có thuộc vâth thể hay không 
class PostProcessSegm(nn.Module):
    # Thiết lập phương thức khởi tạo định nghĩa các thuộc tính 
    def __init__(self, threshold=0.5):
        super().__init__()
        # Định nghĩa một thuộc tính threshold = 0.5. Ngưỡng này sẽ được sử dụng để quyết định 
        # một pixel có thuộc vâth thể hay không 
        self.threshold = threshold

    # sử dụng @torch.no_gradient để bỏ qua việc tính toán gradient cho phương thức
    # tính toán forward của lớp xử lý sau phân đoạn 
    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        # kiểm tra để đản bảo giằng len(org_target_size) == len(max_target_sizes)
        # tức là độ dài của danh sách mục tiêu nguyên bản bằng dộ dài của danh sách tối đa của các mục tiêu 
        assert len(orig_target_sizes) == len(max_target_sizes)
        # Tìm kích thước tối đa từ max_size_target và sử dụng nó để điều chỉnh kích thước của 
        # các mặt nạ dự đoán (predict_mask) thông qua phép nội suy blinear
        max_h , max_w = max_target_sizes.max(0)[0].tolist()
        # gán cho output_mask = predict_mask trong từ điển outputs thực hiện loại bỏ đi 1 chiều có kích thước 
        # bằng 1 được chỉ đinh 
        # từ điển outputs shape =[batch_size , num_class, 1, height, weight ]
        outputs_masks = outputs["pred_masks"].squeeze(2)
        # sau đó thực hiện nội suy cho mặt nạ sao cho nó có kích thước bằng max_sizes 
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        # áp dụng hàm sigmoid cho các mặt nạ và phân bổ chúng cho CPU 
        # áp dụng một ngưỡng threshold để có được mặt nạ nhị phân
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()
        

        # sử dụng một vòng lặp for lấy ra các chỉ số và giá trị các dnah sách ouputs_masks, max_target_sizes, 
        # orig_target_sizes , orig_target_sizes. 
        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            # lấy ra giá trị t[0], t[1] là h và w của ảnh trong trong max_size 
            # mỗi t là 1 tuple trong dnah sách max_target_sizes h và w 
            img_h , img_w = t[0], t[1]
            # cắt danh sahc cur_mask chỉ lấy phần của mặt nạ dự đoán có kích thước tối đa của hình ảnh 
            # lưu vào từ điển results với keys = masks và values là cur_mask thỏa mãn h , w 
            # sử dụng unsqueeze để thêm chiều thứ 2 = 1 vào tensor mask
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            # thực hiện nội suy lên mask cho nó có cùng kích thước với kích thước ban đầu của ảnh 
            results[i]["masks"] = F.interpolate(
                # CHUYỂN ĐỔI CÁC TENSOR MASKS SANG KIỂU DỮ LIỆU FLOAT 
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte() # Chuyển đổi tensor trở lại thành kiểu dữ liệu byte sau khi nội suy.

        # Note : SAU KHI MÔ HÌNH ĐÃ DỰ ĐOÁN CÁC FEATURES MAP CHO PHÂN ĐOẠN , CẦN PHẢI NỘI SUY 
        # CHÚNG CHỞ LẠI KÍCH THƯỚC BAN ĐẦU CỦA ẢNH , ĐỂ SO SÁNH CHÍNH XÁC VỚI GROUND TRUTH HOẶC 
        # HIỂN THỊ MẶT NẠ PHÂN ĐOẠN CHO HÌNH ẢNH GỐC. 
            # Đây là lý do tại sao tt, kích thước ban đầu của hinh anhe được sử dụng để đặt kích thước mới 
            # trong quá trình nội suy 

        # cuối cùng trả về từ điển result 
        return results 
    


# Xây dựng lớp PostProcessPanoptic được thiết kế để chuyển đổi đầu ra của mô hình 
# thành kết quả panoptic cuối cùng theo định dnagj mà cocopanoptic mong đợi 
# là một định dạng coco được xây dựng để xử lý các lớp xử lý như mask , segmentation và image_id 
# cũng nhưu cateforical_id sử dụng cho phân loại đối tượng,từ nguồn dữ liệu gốc 
class PostProcessPanoptic(nn.Module):
    """
    This class converts the output of the model to the final panoptic result, in the format expected by the 
    coco  panoptic API. 
    """
    # Thiết lập phương thức khởi tạo và định nghĩa các tham số thuộc tính 
    # nhận đầu vào gồm Is_thing_map là một từ điển các ID lớp và giá trị là một boolean chỉ 
    # ra lớp đó là thing(True) hay stuff(False)
    # Và Threshold là ngưỡng tin cậy, các phân đoạn có độ tin cậy thấp hơn ngưỡng sẽ bị xóa
    def __init__(self, is_thing_map, thresold=0.85):
        """"
        
        Parameters: 
            is_thing_map: This is whose keys are class IDS, and the values a boolean indicating whether 
                the class is a thing (True) or stuff (False)

            Threshold: Confidence threshold: Segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.thresold = thresold 
        self.is_thing_map = is_thing_map 
    
    # Thiết lập Phương thức forward tính toán dự đoán panoptic từ đầu ra của mô hình.
    # phương thức này sẽ được gọi khi nó nhận được các tham số truyền vào 
    def forward(self, outputs, processed_sizes, target_sizes=None):
        # nhận đầu vào với processed_size là danh sách các tuple(hoặc tensor) chứa 
        # kích thước của hình ảnh sau khi áp dụng tăng cường dữ liệu nhưng trước khi 
        # gộp chúng lại 
        # Target_size là danh sách các tuple(hoặc tensor) tương ứng với kích thước cuối cùng 
        # mong muốn cho mỗi dự đoán. Nếu không được chỉ định nó mặc định là processed_size

        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                            model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                        of each prediction. If left to None, it will default to the processed_sizes
        """

        # kiểm tra target_size 
        if target_size is None: 
            # nếu như danh sách anyf rỗng gán nó = danh sách processed_size 
            target_size = processed_sizes 
        # kiểm tra và đảm bảo giằng 2 danh sách này bằng nhau 
        assert len(processed_sizes) == len(target_size)
        # lấy ra thông tin nhật ký logits , mặt nạ dự đoán và hộp dự đoán từ
        # predl_logits là một tensor hoặc một danh sách các tensor chứa các logit dự đoán . 
        # Mỗi logits biểu diễn mức dộ tin cậy của mô hình rằng pixel cụ thể thuộc vêg một lớp cụ thể 
        # về một lớp. shape = [batch_size , num_queries, num_classes]
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        # và đảm bảo rằng len của các danh sách anyf bằng với len target_size
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        # tạo 1 danh sách pred để lưu trữ các dự đoán 
        preds = []

        # định nghĩa một tuple để chuyển 1 tensor thành 1 tuple 
        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())
        
        # duyệt qua một loạt các dnah sách và gán giá trị tương ứng cho nó 
        for cur_logits, cur_masks, cur_boxes, size , target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # tính toán softmax cho curlogits áp dụng cho chiều cuối cùng 
            # và lấy ra scores và label của nó 
            scores, labels = cur_logits.softmax(-1).max(-1)
            # tạo một phép điều kiện để lọc các gía trị không thỏa mãn , 
            # hàm ne [not equal] sẽ loại bỏ đi các giá trị không thỏa mãn một điều kiện nằm trong nó 
            # ouputs["pred_logits"].shape[-1] - 1 lấy số lượng lớp - 1 để xác định chỉ số lớp nền và điểm scores > 0.85
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            # tương tự như trên áp dụng softmax cho cur_class 
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            # chỉ giữ lại những giá trị phù hợp của cur_scores theo tiêu chuẩn của keep 
            cur_scores = cur_scores[keep]
            # thực hiện tương tự như trên sử dụng lọc kết quả bằng 
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            # sau đó thực hiện phép nội suy lên mặt nạ hiện tại với việc thêm 1 chiều none = 1 vào mặt nạ ban đầu 
            # và size = size tuple này sẽ trả về 2 kích thước h, w theo size 
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            # từ danh sách cur_boxes chứa tọa độ x , y , h , w chuyển nó thành tọa độ hộp  giới hạn 
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            # lấy ra kích thước h và w từ mặt nạ mask shape ban đầu = [num_queries, h , w ]
            h, w = cur_masks.shape[-2:]
            # đảm bảo rằng độ dài của danh sách cur_box và cur_classes bằng nhau 
            assert len(cur_boxes) == len(cur_classes)


            # làm phẳng mặt nạ dự đoán để chúng có thể xử lý dễ dnagf hơn
            cur_maskes = cur_masks.flateten(1)
            # xâyd ựng một từ điển mặc định để theo dõi các ID mặt nạ cho mỗi lớp "stuff". Các mặt ạn stuff
            # cùng lớp với nhau sẽ được gộp lại sau này 
            stuff_equiv_classes = defaultdict(lambda: [])
            # lấy ra chỉ số và nhãn lớp từ danh sách cur_class
            for k, label in enumerate(cur_classes):
                # nếu như nhãn nó không phải is_thing = True 
                if not self.is_thing_map[label.item()]:
                    # thêm nó vào từ điển vừa xâyd ựng 
                    stuff_equiv_classes[label.item()].append(k)

            
            # đingj nghĩa hàm get_ids_area đẻ tạo hình ảnh phân đoạn pannoptic và tính diện 
            # tích của các mặt nạ xuất hiện trên hình ảnh 
            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                # chuyển vị mặt nạ và áp dụng softmax trên chiều cuối cùng 
                # để biến nó thành các xác suất 
                m_id = masks.transpose(0, 1).softmax(-1)

                #kiểm tra xem có phát hiẹn mặt nạ nào không 
                if m_id.shape[-1] == 0:
                    # nếu không có tạo mặt nạ m_id = ma trận zeros shape = h * w
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:# trường hợp có tồn tại
                    # ta reshape h * w và tìm ra chỉ số lớn nhất trong chiều cuối của ma trận 
                    m_id = m_id.argmax(-1).view(h, w)

                # nếu deup = True 
                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    # thực hiện hợp nhất các mặt nạ tương ứng cùng lớp stuff
                    for equiv in stuff_equiv_classes.values():
                        # kiểm tra xem lèn stuff > 1
                        if len(equiv) > 1:
                            # lặp qua từng id mặt nạ trong nhóm 
                            for eq_id in equiv:
                                # thay thế tất cả các pixels trong m_id có giá trị bằng not equal 
                                # id của mặt nạ hiện tại điều này gộp tất acr các mặt nạ cho cùng 1 lớp 
                                # "stuff" thành một mặt nạ duy nhất 
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                
                # lấy ra h và w từ target_size là một tensor
                final_h, final_w = to_tuple(target_size)

                # Chuyển đổi ID mặt nạ thành hình ảnh RGB và thay đổi kích thước của nó để phù hợp với kích thước mục tiêu cuối cùng.
                # 1: chuyển đổi tensor ID thành một mảng numby RGB , m_id.view(h, w) thay đổi hình dạng của tensor 
                # để có size = h */ w
                #  Image.fromarray tạo một đối tượng ảnh từ mảng numpy 
                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                # 2: Thay đổi kích thước của hình ảnh để phù hợp với kích thước mục tiêu cuối cugf 
                # sử dung Image.NEAREST để giữ nguyên các giá trị pixel khi thay đổi kích thước
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)
                
                # 3: seg_img.tobytes(): Chuyển đổi hình ảnh thành chuỗi byte.
                # torch.ByteTensor(...): Tạo một tensor PyTorch từ chuỗi byte.
                # view(final_h, final_w, 3): Thay đổi hình dạng của tensor để có kích thước final_h x final_w x 3, 
                # tương ứng với chiều cao, chiều rộng, và 3 kênh màu RGB.
                # rgb2id (...): Chuyển đổi mảng RGB trở lại thành tensor ID mặt nạ.
                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))
                
                # khởi tạo danh sách để lưu trữ diện tích của mặt ạn 
                area = []
                # vòng lặp for tính toán diện tích của mỗi mặt nạ bằng cách đếm số lượng pixel có ID tương ứng trong tensor m_id.
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                # Trả về danh sách diện tích và hình ảnh phân đoạn đã được thay đổi kích thước.

                #Kết quả là bạn có một danh sách diện tích cho mỗi mặt nạ và một hình ảnh phân đoạn panoptic có kích thước mục tiêu cuối cùng
                return area, seg_img # điều này hữu ích cho viêvj đánh giá chất lượng các phân đoạn 


            #tính toán diện tích phân đoạn và ảnh phân đoạn cho mạt nạ cur và cur_scors
            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            # kiểm tra xem số lượng các phần tử của danh sahcs cur_class > 0 hay có nhiều hơn 1 lớp 
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                # lọc các mặt nạ trống 
                while True:
                    # filtered_small là một tensor boolean được tạo ra từ việc kiểm tra xem diện tích của mỗi mặt nạ có nhỏ hơn hoặc bằng 4 pixel hay không. 
                    # Điều này được thực hiện bằng cách sử dụng một list comprehension, nơi mà area[i] <= 4 kiểm tra điều kiện cho mỗi mặt nạ dựa trên diện tích được tính toán trước đó.
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    #  Kiểm tra xem có bất kỳ mặt nạ nào cần được lọc hay không. Nếu có, thì tiếp tục quá trình lọc.
                    if filtered_small.any().item():
                        # Cập nhật cur_scores bằng cách loại bỏ các điểm số tương ứng với các mặt nạ nhỏ.
                        # sử dụng toán tử ~ để áp dụng lọc lên tất cả các vùng diện tích mặt nạ trong ảnh 
                        cur_scores = cur_scores[~filtered_small]
                        #  Tương tự, cập nhật cur_classes bằng cách loại bỏ các lớp tương ứng với các mặt nạ nhỏ.
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        # Tính toán lại diện tích và hình ảnh phân đoạn sau khi lọc các mặt nạ nhỏ.
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else: #  Nếu không có mặt nạ nào cần được lọc, thoát khỏi vòng lặp.
                        break

            
            else: # tức là  cur_class  <= 0 

                # tạo một tensor curren_class mới được tạo với một phần tử có giá trị là 1. Điều này đảm bảo rằng sẽ luôn có ít nhất một lớp trong cur_classes.
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)
            

            # tạo một dang sách segment_info để lưu trữ thông tin của từng phân đoạn 
            segments_info = []
            # duyệt qua dnah sách diện tích của các masks trong ảnh phân đoạn
            for i, a in enumerate(area):
                # Với mỗi phân đoạn, thông tin bao gồm ID của phân đoạn, xác định nó là “thing” hay “stuff” dựa trên is_thing_map, ID của lớp (category_id),
                # và diện tích của phân đoạn (area).
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            # cur_classes được xóa sau khi thông tin đã được lưu vào segments_info để giải phóng bộ nhớ.
            del cur_classes

            # Sử dụng io.BytesIO() để tạo một luồng byte trong bộ nhớ.
            with io.BytesIO() as out:
                # Lưu hình ảnh phân đoạn (seg_img) vào luồng byte dưới dạng PNG.
                seg_img.save(out, format="PNG")
                # out.getvalue() lấy chuỗi byte PNG từ luồng và lưu vào predictions dưới khóa "png_string".
                # segments_info cũng được thêm vào predictions.
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
                # predictions được thêm vào danh sách preds, chứa tất cả các dự đoán.
            preds.append(predictions)
        
        # return preds trả về danh sách cuối cùng của các dự đoán.
        return preds
