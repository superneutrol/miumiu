import torch 
from torch import nn, einsum 
import torch.nn.functional as F 

import math 
from einops import rearrange 
from config.config import CFG

# khối kết nối dư 
class Residual(nn.Module):
    # Thiết lập phương thức khởi tạo 
    def __init__(self, fn):
        super().__init__()
        self.fn = fn 

    # Thiết lập phương thức forward phương thức này sẽ được gọi khi các 
    # tham số được tryền vào 
    def forward(self, x, **kwargs):
        # trả về phép biến đổi x thông qua mạng neural ánh xạ x 
        return self.fn(x, **kwargs) + x 
    

# Lớp chuẩn hóa trước (bình thường hóa trước)
class PreNorm(nn.Module):
    # thiết lập phương thức khởi tạo 
    def __init__(self, dim, fn):
        super().__init__()
        # định nghĩa một lớp LayerNormalization 
        self.norm = nn.LayerNorm(dim)
        # một lớp feedforward network 
        self.fn = fn 

    # Thiết lập phương thức forward 
    def forward (self, x , **kwargs):
        # trả về kết quả của việc ánh xạ x qua các lớp ffn và layer norm 
        return self.fn(self.norm(x), **kwargs)
    

# Hàm kích hoạt Gelu 
class GEGLU(nn.Module):
    # xây dựng hàm kích hoạt gelu 
    def forward(self, x):
        # sử dụng hàm chunk để chia x thành các phần nhỏ theo chiều biểu diễn 
        x, gates = x.chunk(x , dim =-1)
        # áp dụng hàm kích hoạt Gelu 
        return x * F.gelu(gates)
    

# Xây dựng khối FeedforWard
class FeedForward(nn.Module):
    # Thiết lập phương thức khởi tạo 
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        # định nghĩa một thuộc tính tham số inner 
        inner_dim = int(dim * mult)
        # xâydựng một khối tuần tự 
        self.net = nn.Sequential(
            # một lớp tuyến tính 
            nn.Linear(dim, inner_dim * 2),
            # hàm kích hoạt của Gelu 
            GEGLU(),
            # 1 lớp dropout 
            nn.Dropout(dropout), # optional dropout
            # và lớp ánh xạ tuyến tính cuối cùng 
            nn.Linear(inner_dim, dim)
        )

    
    # phuuwong thức forward trả về thuộc tính net 
    def forward(self, x):
        return self.net(x)
    

# Xây dựng ma trận Bias vị trí tương đối 
class T5RelatedPositionBias(nn.Module):
    # Thiết lập phương thức khởi tạo 
    def __init__(self, scale, num_buckets = 32,  max_distance = 128, heads=8):
        super().__init__()
        # một hệ số tỷ lệ chiều chỉnh 
        self.scale = scale 
        # số lượng buckets để phân loại khoảng cách tương đối 
        self.num_buckets = num_buckets
        # khoảng cahcs tối da có thể 
        self.max_distance = max_distance
        # tạo Embedding cho bias vị trí tương đối , với kích thước num_batch x heads 
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    # Xây dựng một hàm tĩnh để tính toán buckets vị trí tương đối 
    @staticmethod 
    def _realtive_position_bucket(
        relative_position, 
        num_buckets = 32, 
        max_distance = 128 , 
        heads = 8 
    ):
        # đảo ngược vị trí để có giá trị dương 
        n = - relative_position
        # đảm bảo rằng tất cả các giá trị không âm 
        n = torch.max(n, torch.zeros_like(n))

        # Xác định một số lương bucket tối đa 
        max_extract = num_buckets // 2 
        # kiểm tra xem khoảng cách có nhỏ hơn max_extract không 
        is_small = n < max_extract

        # Tính tóan giá trị cho các khoảng cách lớn 
        val_if_large = max_extract + (torch.log(n.float() / max_extract) / math.log(max_distance / max_extract) * (num_buckets - max_extract)).log()
        # giới hạn tensor val_if_large bởi num_buckets - 1 để đảm bảo rằng nó không vượt quá số lượng 
        # buckets tối đa có thể có. bằng cách so sánh 2 tensor cùng kích thước 
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets- 1))
        # sử dụng where để chọn giữa n đối với các khoảng cách nhỏ hơn max_extract và val_if_large 
        # đối với các khoảng cách lớn hơn hoặc bằng max_extract. 
        # nếu is_small = True (khoảng cahcs nhỏ hơn max_extract) sẽ chọn n 
        return torch.where(is_small, n , val_if_large)
    

    # Hàm forward để tính toán bias vị trí tương đối và áp dụng nó vào qk_dót 
    def forward(self, qk_dots):
        # shape -2: lấy ra 2 kích thước cuối cùng 
        i, j, device = *qk_dots.shape[-2:], qk_dots.device 
        # từ 2 kích thước i , j tạo ra2 ma trận q , k position 
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)

        # tính toán vị trí tương đối cho ma trận k và q 
        # sywr dụng hàm rearrager để mở rộng 2 tensor q , k thêm vào nó 
        # 1 chiều shape = 1 vào trước chiều chỉ định 
        # sau đó trừ ma trận cho nhau để có được các vị trí tương đối theo i , j 
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')

        # sau đó xác định bucket vị trí tương đối  hàm _relative_pos_ được gọi với tham số 
        # relpos để xác định bucket tương ứng cho mỗi khoảng cách tương đối 
        # num_buckets và max_distance được sử dụng để xác định số lượng buckets và khoảng cách tối đa có thể.
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        # sử dụng các giá trị bucket từ trên để lấy bias_ tương ứng từ một lớp embedding 
        # Mỗi bucket sẽ được ánh xạ đến một vector bias riêng biệt cho mỗi “head” của cơ chế attention.
        values = self.relative_attention_bias(rp_bucket)
        # rearrange(values, 'i j h -> () h i j') định hình lại tensor bias để phù hợp với kích thước của qk_dots.
        bias = rearrange(values, 'i j h -> () h i j')
        # trả về kết quả cuối cùng ta áp dụng bias được được điều chỉnh tỷ lệ hợp lý 
        # vào qk_dots 
        return qk_dots + (bias * self.scale)


# Thiết lập phương thức Attention 
class Attention(nn.Module):
    # Thiết lập phương thức khỏi tạo và định nghĩa các thuộc tính 
    def __init__(self, *, dim, heads=8, dim_heads = 64, dropout = 0.):
        super().__init__()
        self.heads = heads 
        self.scale = dim_heads ** -0.5  
        inner_dim = heads*  dim_heads

        # dropout layer 
        self.dropout = nn.Dropout(dropout)

        # Tính toán vector q 
        self.to_q = nn.Linear(dim, inner_dim, bias= False)
        # Tính toán vector k và c 
        self.to_kv = nn.Linear(in_features=dim , out_features=dim_heads* 2, bias=False)
        # Và định nghĩa một lớp chiếu tuyến tính 
        self.to_out = nn.Linear(inner_dim, dim)

        # Định nghĩa một relative Position Bias 
        self.rel_pos_bias = T5RelatedPositionBias(scale= dim_heads ** 0.5, heads=heads)

    # Thiết lập phương thức forward phương thức này sẽ được gọi khi có các tham số truyền vào 
    def forward (self, x):
        # lấy ra kích thước numheads, và divice 
        h, device = self.heads, x.device 

        # Tính toán q , k ,v sử dụng hàm chunk cho vector kv để tách vector này thành 2 vector 
        # k , v rơi rạc theo chiều biểu diễn 
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        # thay đổi hình dạng của vector q shape[batch_size, qeg_length , head_dim * num_head] -> shape [batch_size, heads, seq_length, head_dim]
        # và chỉ định h là number header số đầu chú ý 
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        # sau đó sử dụng tỷ lệ scale cho vector q 
        q = q * self.scale

        # Tính tích q. k shape = [batch_size , head_dim , seq_length , seq_length]
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        i, j = sim.shape[-2:] # lấy ra 2 kích thước i , j chính là seq_length để thực hiện xây dựng ma trận bias tương đối 
        # và mặt nạ masks 

        # T5 Relative Positional Bias
        # xây dựng ma trận bias tương đối 
        sim = self.rel_pos_bias(sim)

        # Mặt nạ nhân quả với triu chỉ định chỉ giữ lại phần trên bên phải là 1 tam giác 
        causal_mask = torch.ones(size=(i, j), stype= torch.bool, device = device).triu(j - i + 1)
        # áp dụng masks lên kết quả ma trận vị trí tương đối 
        # thay thế các phần tử của sim tại vị trí được chỉ định bởi causal_mask bằng giá trị -torch.finfo(sim.dtype).max.
        # Điều này đảm bảo rằng các phần tử bị mask sẽ không ảnh hưởng đến kết quả cuối cùng.
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # Tính toán softmax cho kết quả {q*k / dk ** 0.5} + {m , b} 
        attn = sim.softmax(dim=-1)
        # áp dụng một lớp dropout attention cho các giá trị không thỏa mãn 
        attn = self.dropout(attn)

        # chuyển vị tensor attn và tính toán attention score  qk*v 
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # chuyển đổi tensor out thành 1 tensor shape =[batch_size, seq_length, h*d]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # áp dụng một lớp tuyến tính cho dữ liệu tăng cường tuyến tính cho các tham số 
        return self.to_out(out) # trả về kết quả là một danh sách attention score
    


# Xây dựng khơi Transformer 
class Transformer(nn.Module):
    # định nghĩa một phuuwong thức khởi tạo 
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # duyệt qua số lượng depth (là số lượng lớp đầu chú ý ) 
        for _ in range(depth):
            # tại mỗi bước thời gian thực hiện một danh sách các quy trình 
            self.layers.append(nn.ModuleList([
                # một bộ phận chuẩn hóa kết quả tính toán chú ý
                Residual(PreNorm(dim, Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                # và bộ phận chuẩn hóa đầu ra chú ý thông qua feedforward network 
                Residual(PreNorm(dim, FeedForward(dim = dim, dropout = dropout)))
            ]))
    # thiết lập phương thức forward, phuuwong thức này sẽ được gọi khi các tham số 
    # được chuyền vào 
    def forward(self, x):
        # duyệt qua danh sách layer lớp đầu chú ý 
        for attn, ff in self.layers:
            # áp dụng tính toán attention cho đầu vào 
            x = attn(x)
            # sau đó chuyển tiếp dữ liệu qua mạng ffn 
            x = ff(x)
        # trả về kết quả cuối cùng là đầu ra của khối transformer 
        return x


# Xây dựng mô hình LamDA 
class LaMDA(nn.Module):
    # Thiết lập phuuwong thức khởi tạo 
    def __init__(self, *, num_tokens, dim , depth, dim_head, heads): 
        super().__init__()
        # nhúng các tokens sử dụng lớp nhúng Embedding 
        self.token_dim = nn.Embedding(num_embeddings=num_tokens, embedding_dim=dim)

        # Xây dựng khối xử lý tranformer 
        self.transformer = Transformer(dim, depth, dim_head, heads)
        
        # khối chuẩn hóa và ánh xạ tuyến tính dữ liệu tham số 
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    # Thiết lập phương thức forward 
    def forward(self, x):
        # thực hiện nhúng trước 
        x = self.token_emb(x)
        # chuyển tiếp vào model transformer 
        x = self.transformer(x)
        # áp dụng lớp chuẩn háo và ánh xạ tuyến tính dữ liệu 
        logits = self.to_logits(x)
        # trả về kết quả cuối cùng 
        return logits
    

# Cấu hình cho mô hình ngôn ngữ LaMDA 
def lamda_model():
    model = LaMDA(
        # TÍNH TOÁN SỐ LƯỢNG TOKEN CHO MỘT CHUỖI ĐẦU VÀO 
        num_tokens = CFG.num_tokens,
        # EMBED_DIM 
        dim = CFG.dim,
        # DEPTH SỐ LƯỢNG LỚP CHÚ Ý 
        depth = CFG.depth,
        # KÍCH THƯỚC MỖI ĐẦU 
        dim_head = CFG.dim_head, 
        # SỐ LƯỢNG ĐẦU
        heads = CFG.heads

    )

# KIỂM TRA XEM CÓ ĐANG HOẠT ĐỘNG Ở TRƯƠNG TRÌNH CHÍNH 
if __name__ == "__main__":
    # LaMDA model 
    lamda_base = lamda_model()

    #lamda = AutoregressiveWrapper(lamda_base, max_seq_len = 2048)

    tokens = torch.randint(0, 20000, (1, 2048)) # mock token data


    logits = lamda_base(tokens)
    print(logits.shape)

    n_params_torch = sum(
        p.numel() for p in lamda_base.parameters() if p.requires_grad
    )

    print(f"Number of parameters in torch model: {n_params_torch}")