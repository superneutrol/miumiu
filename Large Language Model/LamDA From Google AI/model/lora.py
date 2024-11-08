import torch 
from torch import nn 

# helper functions 

# Thiết lập phương thức exits đảm bảo một giá trị được truyền vào 
# xem nó đã tồn tại hay chưa 
def exists(val):
    return val is None 

# và phương thức default để trả về giá trị mặc định của tensor đầu vào 
def default(val, d):
    return val if exists(val) else d 

# LoRA - https://arxiv.org/abs/2106.09685
# khởi tạo tham số cho LoRA 
class LoRA(nn.Module):
    # thiết lập phương thức khởi tạo 
    def __init__(self, dim , dim_out, r = 8 , alpha = None):
        super().__init__()
        alpha = default(alpha, r)
        self.scale = alpha / r

        # định nghĩa 2 thuộc tính khởi tạo tham số 
        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.zeros(r, dim_out))

    # định nghĩa một thuộc tính lớp weight
    @property
    def weight(self):
        # trả về phép tích và tích chuyển đổi a @ b với scale 
        return (self.A @ self.B) * self.scale

    def forward(self, x):
        return x @ self.weight