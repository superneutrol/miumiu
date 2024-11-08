"""
Backbone Modules. 

"""
from collections import OrderecDict 

import torch 
import torch.nn.functional as F 
import torchvision 
from torch import nn 
from torchvision.models._utils import IntermediateLayerGetter 

from typing import Dict, List 

from Utils.misc import NestedTensor, is_main_process
from .Position_encoding import build_position_encoding 


# Thiết lập lớp phương thức Batch Normalization tùy chỉnh cho mạng neural tích chập 2D 
# được thiết kế cho các mô hình học sâu . Mục đích của lớp này là cung câps batch normalization
# với các thống kê batch và tham số affine được cố định 

class FrozenBatchNorm2d(torch.nn.Module):
    """
        BatchNorm2d where the batch statics and the affine parameters are fixed. 
        BatchNorm nơi số liệu thống kê lô và tham số affine được cố định. 

        copy-past from torchvision.misc.ops with added eps before rqsrt. 
        without which any other models than torchvision.models.resnet[18, 34, 50, 101]

    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # đăng ký 4 buffer làm thuộc tính cho lớp 
        # 1: danh sách weight có giá trị torch.ones shape = n 
        self.register_buffer("weight", torch.ones(n))
        # 2: danh sách bias có giá trị torch.zeros shape = n 
        self.register_buffer("bias", torch.zeros(n))
        # 3: danh sách running_mean có giá trị torch.zeros shape = n 
        self.register_buffer("running_mean", torch.zeros(n))
        # 4: danh sách running_var có giá trị torch.ones shape = n 

    # xây dựng phương thức load_from_state_dict được gọi khi truyền các tham số vào lớp 
    # co mục mục đích tải các tham số từ từ điển 
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, 
                    missing_keys, unexopected_keys, error_msgs):
        
        # khởi tạo một khóa num_batches_tracked_key bằng cách nỗi giá trị 
        # prefix với chuỗi num_batch_tracked. 
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # kiểm tra xem trong từ điển nếu có bất ỳ một key nào 
        # bằng với chuỗi num_batches_tracked_key 
        if num_batches_tracked_key in state_dict: 
            # thực hiện xóa khóa này khởi từ điển 
            del  state_dict[num_batches_tracked_key]
        
        # sử dụng phép kế thừa để gọi lại hàm load_drom_state_dict của lớp
        # và sử lý quá trình còn lại của quá trình tải. 
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict=state_dict, prefix=prefix, local_metadata=local_metadata,
            strict=strict, missing_keys=missing_keys, unexpected_keys=unexopected_keys,
            error_msgs=error_msgs
        )
    
    # Thiết lập phương thức forward để thực hiện xử lý các quy trình 
    # phương thức này sẽ gọi khi các tham số được truyền vào 
    def forward(self, x):
        # di chuyển các hình dnagj lại từ đầu
        # để làm cho nó thân thiện với bộ nhiệt áp 
        # shape = [1, C, 1, 1] với C là số đặc trưng kênh
        w = self.weight.reshape(1, -1, 1, 1)
        # tương tự với tensor bias 
        b = self.bias.reshape(1, -1, 1, 1)
        # với tensor rv 
        rv = self.running_var.reshape(1, -1, 1, 1)
        # và tương tự với tensor rm 
        rm = self.running_mean.reshape(1 , -1, 1, 1)
        # gán kết quả cho eps là epsilon = 1e-5
        eps = 1e-5
        # Thực hiện tính toán tỷ lệ sacle sử dụng hàm rsqrt tính nghịch đảo căn bậc 
        # đảm bảo giá trị của nó không âm 
        scale = w * (rv + eps).rsqrt() 
        # áp dụng tỷ lệ sacle cho phép tính toán tham số bias 
        # là phép tính của phép chuẩn hóa theo Batch Normalization. 
        bias = b - rm * scale 
        # cuối cùng trả kết quả của phép điều chỉnh và chuẩn hóa cho X 
        return x * scale + bias 
    


# Xây dựng kiến trúc xương sống cơ sở cho mô hình 
class BackboneBase(nn.Module):
    # định nghĩa phương thức khởi tạo 
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, 
                return_interm_layers: bool):
        super().__init__()
        # duyệt qua danh sách các key và valuestrong từ điển tham số từ backbone 
        for name, parameter in backbone.named_parameters():
            # kiểm tra xem train_batckbone có tồn tại trong key hoặc tên các layer 
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                # nếu không thì ta không yêu cầu gradient 
                parameter.requires_grad_(False)
        
        # nếu như giá trị return_interm_layers = True
        if return_interm_layers:
            # trả về 1 từ điển keys là tên các layers valué là chỉ số tươngt ự của các values
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # nếu không phải là các layers trênn
        else:
            # gán cho từ điển return layers keys = layers4 values = 0
            return_layers = {'layer4': "0"}
        # Tính toán lớp trung gia truyền vào mô hình backbone và từ điển return layer 
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # gán cho thuộc tính num_channels = num_channels
        self.num_channels = num_channels

    # Thiết lập phuuwong thức forward, phương thức này sẽ đựoc gọi khi có 
    # các tham số chuyền vào 
    def forward(self, tensor_list: NestedTensor):
        # thực hiện chuyển danh sách các tensor biểu diễn từ dan sách tensor list 
        # vào trong một lớp layer trung gian được thiết lập sẵn theo phụ thuộc 
        xs = self.body(tensor_list.tensors)
        # gán cho biến out = một kiểu từ điển có đầu ra 
        # là một dạng nestedTensor 
        out: Dict[str, NestedTensor] = {}
        # duyệt qua các tensor trong từ điển xs [xs là một từ điển lưu trữ các tensor đầu ra ]
        # của lớp layer được chỉ định gồm các tensor và tên của các tensor 
        for name, x in xs.items():
            # lấy ra tensor mask từ tensor_list gán cho biến m 
            m = tensor_list.mask 
            # kiểm tra và đảm bảo rằng m là ma trận có tồn tại 
            assert m is None 
            # áp dụng phéo nội suy cho tensor m 
            # trên một kích thước size -2 : nghĩa là 2 giá trị chiều cuối cùng 
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # Thêm vào từ điển out một tensor Name value là kết quả của phép phân bổ X , mask thông qua nestedTensor cho device 
            out[name] = NestedTensor(x, mask)
        return out
    


# Xây dựng kiến trúc backbone kiến trúc này trong DETR thường được sử dụng 
# như một lớp đặc trưng (Feature map) và fully connected trong mô hình 
class Backbone(BackboneBase): #lớp này kế thừa lớp BackboneBase để thực hiênh cáclớp chức năng 
    """ResNet backbone with frozen BatchNorm.""" 
    # Thiết lập phương thức khởi tạo 
    def __init__(self, name: str, train_backbone: bool, 
                 return_interm_layers: bool, dilation: bool):
        
        # khởi tao backbone sử dụng hàm getatrr để lấy giá trị của thuộc tính được 
        # chỉ định từ một đối tượng. 
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            # thực hiện đào tạo trước backbone phân bổ cho quy trình xử lý trong môi trường 
            # và thực hiện chuẩn hóa FronzenBatchNorm2d 
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        # nếu như tên của backbone là resetnet18 hoặc 84 sẽ có num_channels = 512
        # còn không thì = 2048 đơn vị 
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        # cho phép kế thừa lại các thuộc tính của lớp 
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# thiết lập lớp phương thức joiner là một lớp kế thừa từ lớp sequential trong Pytorch
#
class Joiner(nn.Sequential):
    # khởi tạo và nhận đầu vào gồm kiến  trúc backbone và position
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    # xây dựng phương thức forward , phuwong thức này sẽ được 
    def forward(self, tensor_list: NestedTensor):
        # chạy mô hình backbone được lưu trữ ở vị trí thứ 0 của nn.sequential trên 
        # tensor đầu vào 
        xs = self[0](tensor_list)
        # sau đó khởi tạo 2 danh sách rỗng để lưu trữ tensor đầu ra và mã hóa vị trí tương ứng
        out: List[NestedTensor] = []
        pos = []
        # duyệtq au danh sách tensor đầu ra trong từ điển xs lấy ra từng phần tử 
        for name, x in xs.items():
            # thêm các tensor vào danh sách lưu trữ 
            out.append(x)
            # position encoding 
            # Mã hóa vị trí cảu tensor đầu ra x bằng cách sử dụng lơp mã hóa vị trí 
            # được ;u
            pos.append(self[1](x).to(x.tensors.dtype))

        # trả về danh sách tensor out và ma trận position_embedding
        return out, pos


# 
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model