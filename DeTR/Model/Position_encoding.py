"""Mã hóa vị trí khác nhau cho mô hình máy biến áp
Various positional encodings for the transformer. 
"""

import math 
import torch 
from torch import nn 
from Utils.misc import NestedTensor 

# Thiết lập lớp phương thức PositionEmbeddingSine để tạo ra các nhúng vị trí dựa 
# trên hàm sin và cos khá giống với nhúng vị trí được sử dụng trong Transformer 
# nhưng được tổng quát để hoạt động trên hình ảnh 
class PositionEmbeddingSine(nn.Module):
    """This is a more standard vesion of the position embedding, very similary tp the one
       used by the Attention is all you need paper, generalized to work on images.
    """
    # thiết lập phương thức khởi tạo và định nghĩa các tham số 
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        # định nghĩa num_pos_feats là số lượng các đặc chưng vị trí 
        # cho các patch đầu vào 
        self.num_pos_feats = num_pos_feats
        # và temprerature được sử dụng để điều chỉnh các đặc trưng vị trí 
        self.temperature = temperature 
        # normalized quyết định xem có nên chuẩn hóa các nhúng vị trí hay không 
        self.normalize = normalize
        # Kiểm tra một điểm kiện nếu scale != None và normalize = False 
        if scale is not None and normalize is False : 
            # nếu như điều kiện này đúng thì ném ra một cảnh báo 
            raise ValueError("normalize should be True if scale is passed")
        # Và nếu như scale is None 
        if scale is None: 
            # gán scale = 2 pi 
            scale = 2 * math.pi 
        # gán thuộc tính scale  = biến scale 
        self.scale = scale 

    # Thiết lập phương thức forward để chuyển tiếp thực hiện chức năng có nhiệm vụ 
    # tạo ra các nhúng vị trí 
    # 1 : Tạo ra các nhúng vị trí dựa trên tổng tích lũy của mặt nạ. 
    # 2 : Nếu normalze là True nó sẽ chuẩn hóa các nhúng vị trí. 
    # 3 : Cuối cùng nó tạo ra các nhúng vị trí cuối cùng bằng cách sử dụng hàm sin cos 
    # trên các vị trí nhúng vị trí đã được chuẩn hóa điều chỉnh tỷ lệ
    # 1 ; 2 : Chỉ là thực hiện chuẩn hóa cho việc nhúng vị trí sin cos 
    def forward(self, tensor_list: NestedTensor):
        # đầu vào là danh sách tensor_list một dạng nestedTensor (tensor lồng nhau)
        # 1 : Từ danh sách tensor lấy ra tensor biểu diễn đầu vào 
        x = tensor_list.tensors 
        # 2 : Tương tự lấy ra tensor mask là mặt nạ cho tensor dữ liệu gán nó cho mask 
        mask = tensor_list.mask 
        # kiểm tra một điều kiện và đảm bảo giằng mask != None 
        assert mask is not None
        # lấy phủ định của mask với toán tử ~ chuyển đổi các phần tử 0 <-> 1 cho nhau 
        not_mask = ~mask 
        # thực hiện tính tônge tích lũy với hàm cumsum not_mask 
        # theo 2 chiều ngang dọc 
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # kiểm tra xem giá trị normalize có tồn tại 
        if self.normalize: 
            # gán cho biến epsilon = 1e-6 
            eps = 1e-6 
            # thực hiệnc huẩn hóa các giá trị cuối cùng của hàng trong ma trận embed
            # tức là giá trị lớn nhất theo phép tính tổng tích lũy 
            y_embed =  y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # tương tự với ma rận x_embed là giá trị lớn nhất theo chiều cột 
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # tạo ra một tensor mới có giá trị 0 -? self.num_pos_feats - 1 
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # thực hiện phép tính toán tỷ lệ dim_t
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # chia các tỷ lệ nhúng vị trí cho dim_t
        # chia các nhúng vị trí trong ma trận pos_x 
        pos_x = x_embed[:, :, :, None] / dim_t
        # và ma trận pos_y cho dim_t kết quả của 2 ma trân nầy là một chuỗi các chỉ số giảm dần từ 1 đến 
        # giá trị nhỏ nhất với mỗi đặc trưng vị trí sẽ được biểu diễn theo một tỷ lệ khác nhau
        pos_y = y_embed[:, :, :, None] / dim_t

        # sau đó áp dụng hàm sin và cos lên ma trận pos_x và pos_y kết quả là một tensor mới với các giá 
        # trị sin cos xem kẽ nhau
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # với [0::2] lấy vị trí chẵn bắt đầu từ 0 và 1::2 lấy các vị trí lẽ bắt đầu từ 1 [2 là đơn vị khoảng cách]
        # với flatten(3) sẽ làm phẳng tensor này từ chiều thứ 3 trở đi 
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # nối 2 tensor này theo chiều thứ 3 và sau đó đảo chiều tensor này 
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos



# Xây dựng phương thức lớp PositionEmbedding Learned 
# sử dụng để tạo ra các nhúng vị trí học được trong khi thực nghiệm mô hình 
class PositionEmbeddingLearned(nn.Module):
    """
        Absolute pos embedding, learned.
    """
    # Đingh nghĩa phương thức khởi tạo và định hình các thuộc tính tham số 
    def __init__(self, num_pos_feats=256):
        super().__init__()
        # thiết lập 2 lớp nhúng là row và col 
        self.row_embed = nn.Embedding(num_embeddings=50, embedding_dim=num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        # và đồng thời định nghĩa phương thức reset_parmeter trong hàm khởi tạo 
        # để có thể truy suất nó như một thuộc tính lớp 
        self.reset_parameters()

    # định nghĩa phương thức reset_parameters được sử dụng để khởi tạo lại 
    # các tham số của lớp nhúng theo phân phối 
    def reset_parameters(self):
        # sử dụng một hàm phân phối uniform với tham số nhúng row và col 
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    # Thiết lập phương thức forward của lớp được gọi khi chuyền một đầu vào qua lớp 
    def forward(self, tensor_list : NestedTensor):
        # lấy ra tensor biểu diễn đầu vào từ tensor list 
        x = tensor_list.tensor 
        # lấy ra h w là 2 kích thước cuối cùng của x 
        # là kích thước chiều cao và chiều rộng của hình ảnh 
        h , w = x.shape[-2:]

        # sau đó tạo ra 2 tensor i và j có nhiều giá trị chứa các số 0-> h-1 và w-1
        i = torch.arange(w, device=x.device)
        # thực hiện tạo tensor j 
        # 2 tensor này được sử lý bởi thiết bị tham gia môi trường phân tán 
        j = torch.arange(h, device=x.device)
        # thực hiện nhúng tham số cho ma trận i và j nhúng cho mỗi vị trí hàng và cột 
        # từ các lớp nhúng tương ứng
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        # ghép các nhúng hàng và cột lại với nhau . sau đó hoán đổi chiều tensor 
        pos = torch.cat([
            # thêm 1 chiều mới vào tensor với unsqueeze và lặp lại 
            # tensor dọc theo chiều đó để xác định kích thước của nó 
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
            # sau đó thực hiện nối chúng theo chiều cuối cùng 
            # tương tự như trên ta áp dụng biến đổi vị trí và thêm chiều mới vào tensor kết quả
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # cuối cùng là trả về tensor kết quả là ma trận biểu diễn nhúng mà các vị trí có thể học được
        return pos
    

# xây dựng phương thức buil encoding để xác định kiểu nhúng phù hợp cho kiểu model  
def build_position_encoding(args):
    # xác định 1 giá trị lặp là n_steps 
    N_steps = args.hidden_dim // 2
    # kiểm tra giá trị tham số args trong từ điển tham số của mô hình 
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    # tương tự nếu như là V3 hoặc learned cho tham số pos 
    elif args.position_embedding in ('v3', 'learned'):
        # ta thực hiện áp dụng pos learned cho mô hình 
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        # nếu không nằm trong 2 trường hợp được so sánh ném ra một cảnh báo 
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding