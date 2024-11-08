import math 
from dataclasses import dataclass 
from typing import Optional, Tuple 

import fairscale.nn.model_parallel.initialize  as fs_init 
import torch 
import torch.nn.functional as F 
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn 



# Xây dựng một @dataclass từ thư viện dataclasses trong python 
# @dataclass là một decorator giúp tự động tạo ra các phương thức đặc biệt như __init__
# decorator này chủ yếu được sử dụng để lưu trữ dữ liệu 
@dataclass
class ModelArgs:
    # kích thước số chiều biểu diễn embed_dim = 4096 
    dim: int = 4096
    # number_layer = 32
    n_layers: int = 32
    # num_ber header 32 
    n_heads: int = 32
    # SỐ LƯỢNG HEAD CHO KEY/VALUE TRONG SELF-ATTENTION CÓ THỂ KHÔNG ĐƯỢC ĐẶT 
    n_kv_heads: Optional[int] = None
    # KÍCH THƯỚC CỦA TẬP TỪ VỰNG, -1 ĐẠI DIỆN CHO GIÁ TRỊ CHƯA ĐƯỢC ĐẶT 
    vocab_size: int = -1
    # # Đảm bảo kích thước của lớp ẩn SwiGLU là bội số của một lũy thừa lớn của 2.
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    # KÍCH THƯỚC CHO LỚP FFN 
    ffn_dim_multiplier: Optional[float] = None
    # # Epsilon sử dụng trong Layer Normalization để tránh chia cho 0.
    norm_eps: float = 1e-5
    # # Một tham số cụ thể, có thể liên quan đến RPE (Relative Positional Encoding).
    rope_theta: float = 500000
    #  Kích thước lô tối đa cho việc huấn luyện.
    max_batch_size: int = 32
    # # Độ dài chuỗi tối đa mà mô hình có thể xử lý
    max_seq_len: int = 2048



# xÂY DỰNG lớp chuẩn hóa RMSNorm 
class RMSNorm(torch.nn.Module):
    # Xây dựng phương thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # định nghĩa epsilon 
        self.eps = eps
        # và định nghĩa một trọng số khởi tạo 
        self.weight = nn.Parameter(torch.ones(dim))

    # Xây dựng phương thức norm đây là một phương thức chuẩn hóa 
    def _norm(self, x):
        # trả về kết quả của x được chuẩn hóa 
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    # Phương thức forward sẽ được gọi khi co các tham số truyền vào 
    def forward(self, x):
        # áp dụng hàm self._norm cho x sau đó chuyển kiểu dữ liệu của x thành float 
        output = self._norm(x.float()).type_as(x)
        # cuối cùng nhân kết quả chuẩn hóa với trọng số được khơi tạo 
        return output * self.weight
    


# Xây dựng hàm precompute_freqs_cis được sử dụng để tinh tonas các tần số 
# trước cho mã hóa vị trí 
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # Tính toán các tần số dựa trên công thức mã háo vị trí 
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim())) # Theta 

    # Tạo một tensor chứa các giá trị từ 0 đến end - 1
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # Tính toán tích ngoài của t và freqs để tạo được một ma trận
    # tích ngoài của 2 vector tạo ra một vector mới có các tính chất đặc biệt 
    # Gồm độ lớn. Độ lớn của vector kết quả bằng diện tích của hình bình hành được tạo bởi 
    # 2 vector ban đầu 
    # Hướng. Vector kết quả vuông góc với mặt phẳng chứa 2 vector ban đầu. Hướng của nó được xác định 
    # bởi quy tắc bàn tay phải
    # Phản giao hoán: Tích ngoài của vector a và vector b là -(tích ngoài của nó sẽ khác so với không gian ba chiều)
    freqs = torch.outer(t, freqs) 

    # sử dụng hàm torch.polar để tạo ra một tensor phức hợp từ ácc giá trị tần số 
    # # torch.ones_like(freqs) tạo ra một tensor có cùng kích thước với 'freqs' nhưng tất cả các phần tử đều là 1.
    # 'freqs' đại diện cho phần góc của số phức, và torch.ones_like(freqs) đại diện cho phần độ lớn.
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis



# Xây dựng hàm reshape_for_broadcast trong đoạn mã Python được sử dụng để chuẩn bị 
# tensor Freqs_cis cho quá trình broadcasting với tensor x. Broadcasting là một kỹ thuật 
# trong numpy và Pytorch cho phép thực hiện các phép toán trên các tensor có kich thước khác nhau 
# mà không cần phải saoc hép dữ liệu một cách không cần thiết 
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # Lấy ra số chiều của tebssor x 
    ndim = x.ndim 

    # đảm bảo rằng tensor x có ít nhất 2 chiều 
    assert 0 <= 1 < ndim 

    # Kiểm tra xem kích thước của freqs_cis có phù hợp với chiều thứ 2 và chiều cuối cùng 
    # của x hay không 
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # Tạo một danh sách mới cho kích thước mà tensor freqs_cis sẽ được coi như có 
    # Nếu như i là 1 hoặc là chiều cuối cùng của x, giữ nguyên kích thước tương ứng từ x 
    # Nếu không, đặt kích thước là 1 để cho phép broadcasting 
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    # Sử dụng phương thức view để thay đổi kích thước của freqs_cis mà không thay đổi dữ liệu 
    return freqs_cis.view(*shape)



# Xây dựng phương thức apply_rotary_emb phương thức này là một phần của quá trình áp dụng 
# mã hóa vị trí xoay (rotary position encoding) cho các tensor đầu vào, thường được sử dụng 
# để cải thiện việc xử lý thông tin vị trí 
def apply_rotary_emb(
        xq: torch.Tensor, 
        xk: torch.Tensor, 
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # chuyển đổi tensor xq và xk thành dạng số phức 
    # reshape(*xq.shape[:-1], -1, 2) phương thức này được sử dụng để thay đổi hình dnagj của tensor 
    # xq mà không thay đổi dữ liệu của nó nó cho phép giữ nguyên tất cả các chiều trừ chiều cuối cùng 
    # tham số -1 cho phép tự động tính toán kích thước  của chiều đó dựa trên kích thước tổng thể của tensor và 
    # các chiều khác. Tham số 2 đặt kích thước của chiều cuối cùng là 2, đại diện cho phần thực và phần ảo của số phức.

    # tách chiều cuối cùng thành 2 chiều và tính toán góc xoay theta dựa trên vị trí (p) của vector token x và chỉ số 
    # (i) của chiều d / 2 theo công thức sinusoidal cố định. Sau đó hàm view_as_complex sẽ áp dụng công thức
    # e iθ k ​ = cos(θ k ​) + i.sin(θ k ​)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # tương tự như trên với XK 
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Chuẩn bị freqs_cis cho broadcasting với xq_.
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Nhân từng phần tử của xq_ và xk_ với freqs_cis và chuyển kết quả về dạng số thực.
    #  tensor freqs_cis sẽ chuyển các kết quả 
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # Trả về kết quả với kiểu dữ liệu ban đầu của xq và xk.
    return xq_out.type_as(xq), xk_out.type_as(xk)



# Xây dựng phương thức repeat_kv phương thức này có chức năng lặp lại các
# giá trị của tensor x theo số lần nhất định n_rep 
#  Nhận một tensor x làm đầu vào và n_rep là số lần lặp cho mỗi giá trị trong tensor 
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # Lấy ra các kích thước của tensor x 
    bs, slen, n_kv_heads, head_dim = x.shape
    # nếu n_rep = 1 
    if n_rep == 1:
        # lập tức trả về tensor x 
        return x
    # Nếu n_rep lớn hơn 1
    return (
        # mở rộng tensor x thêm 1 chiều là chiều thứ 4 shape = 1
        x[:, :, :, None, :]
        # sử dụng hàm expand để lặp lại tensor này với chiều thứ 4 có kích thuơc n_rep
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        # sau đó reshape tensor này thành tensor 4 chiều 
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )



# Xây dựng lớp Attention thực hiện tính Attention scores cho các vector QKV 
class Attention(nn.Module):
    # Xây dựng phương thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Num_kv_head thuộc tính này đại diện cho số lượng vector j , v 
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Định nghĩa kích thước mô hình phân tích
        model_parellel_size = fs_init.get_data_parallel_world_size()
        # số lượng head đựoc phân bổ cho mỗi phần của model_parellelism 
        self.n_local_heads = args.n_heads // model_parellel_size
        # số lượng vector k , v cho mỗi phần model parallism 
        self.n_local_kv_heads = self.n_kv_heads // model_parellel_size
        # N_rep tỷ lệ giữa số lượnglocal_head và số lượng local_kv_head 
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # kích thước head_dim  kích thước cho mỗi đầu 
        self.head_dim = args.dim // args.n_heads


        # Mục đích: Phân chia một mô hình hoặc một phần của mô hình (như một lớp nn.Linear hoặc nn.Embedding)
        # theo chiều cột để có thể phân tán trên nhiều GPU1.
        self.wq = ColumnParallelLinear(
            in_features=args.dim, 
            out_features=args.n_heads * self.head_dim , 
            bias = False , 
            gather_output=False, 
            init_method= lambda x : x,
        )
        # Khi sử dụng ColumnParallelLinear, ma trận trọng số của lớp được chia nhỏ 
        # theo chiều cột và mỗi phần được đặt trên một GPU khác nhau. 
        # Điều này giúp giảm bộ nhớ cần thiết cho mỗi GPU và cho phép huấn luyện các mô hình lớn hơn2.
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # Ứng dụng: Thường được sử dụng trong các mô hình có kiến trúc phức tạp như Transformer, 
        # nơi cần phân tán các tính toán liên quan đến ma trận trọng số.
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # Mục đích: Tương tự như ColumnParallelLinear, nhưng thay vì phân chia theo chiều cột,
        #  RowParallelLinear phân chia ma trận trọng số theo chiều hàng.
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # xây dựng bộ nhớ cache cho vector k là một tensor zeros 
        # shape batch_size, seq_length, num_local_kv_heads, self.head_dim 
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        # Tương tự tạo một bộ nhớ  đệm cho vector v 
        self.cache_v = torch.zeros(
            (
                # shape = [batch_size, max_seq_len, num_local_kv_heads, head_dim]
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda() # sau đó chuyển việc tính toán cho cuda 
    
    # Thiết lập phương thức forward phương thức này sẽ được gọi khi có các tham số 
    # được truyền vào: Phương thức này nhận đầu vào gồm Tensor đầu vào , chỉ số vị trí start_pos 
    # tensor tần suất của token freqs_cis và mặt nạ mask 
    def forward(self, x: torch.Tensor, start_pos: int, 
            freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] ):
        # lất ra các kích thước của tensor đầu vào 
        bsz, seqlen , _ = x.shape 
        # Thực hiện nhúng tuyến tính để có được các tensor q, k, v 
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Thực hiện reshape tensor xq về hình dạng [batch_size, seq_len, num_local_head, head_dim]
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # Và reshape các tensor k, v về dạng shape [batch_Suze, seq_len, num_local_kv_head, head_dim]
        # với number_local_kv_heads là số lượng vector k , v được phân phối cho mỗi phần 
        # của mô hình song song 
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # áp dụnng nhúng xoay chiều lên vector xq và xk  với tần xuất freqs_cis 
        # được thiết lập 
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # chuyển kết quả của k và v sang bộ nhớ cache 
        # và thực hiện trên chúng 1 lát cắt phần lát cắt này sẽ gán cho kết 
        # quả của k và v tương ứng 
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # Cắt vector keys và values theo lát cắt từ bộ nhớ cache để lấy ra 
        # kích thước chứa phần biểu diễn tương ứng 
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        # Hàm rêpat được gọi để lặp tensor k và mở rộng nó thành 1 tensor có 
        # 4 chiều 
        keys = repeat_kv(
            keys, self.n_rep
            # với cache len là độ dài hiện tại được lưu trữ trong cache và seq_len ở đây sẽ là độ dài đã đựoc
            # lưu trữ trước đó 
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # thực hiện chuyển vị các tensor Q K V 
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # thực hiện tính tích vector Q.K.T / sqrt(dk)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # kiểm tra xem mặt nạ mask có được xây dựng chưa
        if mask is not None:
            # nếu mask tồn tại cộng mask vào kết quả của attention_scores 
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # hàm softmax được gọi để áp dụng lên tích của Q.K.T / SQRT(DK) + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # Tích tích giưa vector V với attention scores 
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # hàm countigous được sử dụng để đảm bảo dữ liệu vẫn sẽ được lưu trữ liên tục trong bộ 
        # nhớ và chuẩn bị dữ liệu để thực hiện cho phép toán tiếp theo
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
    


# Xây dựng lớp mạng chuyển tiếp dữ liệu Feedforward NetWork 
class FeedForward(nn.Module):
    # Thiết lập phương thức khởi tạo và đonhj nghĩa các thuộc tính 
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # Tính toán hiddden_dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        # kiểm tra xem ffn_dim có được thiết lập hay không 
        if ffn_dim_multiplier is not None:
            # Nếu chưa ta cần thiết lập nó 
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # định nghĩa w1 sử dụng ColumParallelLinear để tách tuyến tính tensor đầu vào thành các cột
        # có kích thước như nhau và mỗi phần sẽ đặt trên một GPU khác nhau việc này đảm bảo 
        # tăng hiệu quả và giảm đi thời hian tính toán 
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        # thuộc tính w2 khi được áp lên tensor nó sẽ tách các tensor thành các hàng và chueyenr 
        # các phần riêng lẻ cho các GPU khác nhau 
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        # thuộc tính w3 tương tự như W1
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    # Phương thức forward sẽ được gọi khi có các tham số truyền vào
    def forward(self, x):
        # áp dụng các thuộc tính lên tensor đầu vào 
        # áp dụng hàm sigmoid Linear lên kết quả 
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Xây dựng khối transformer thực hiện 
# xây dựng nên một kết cấu của 1 khối transformer cơ bản 
class TransformerBlock(nn.Module):
    # Thiết lập ohuowng thức khưởi tạo và định nghĩa các thuộc tính 
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        # thuộc tính heads 
        self.n_heads = args.n_heads
        # kích thước nhúngb
        self.dim = args.dim
        # kích thước mỗi đầu 
        self.head_dim = args.dim // args.n_heads
        # lớp attention 
        self.attention = Attention(args)
        # MẠNG FEEDFORWARD network 
        self.feed_forward = FeedForward(
            # lớp này nhận các tham số 
            dim=args.dim,
            # đầu ra , đầu mở rôngk
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        # num_ber_layer 
        self.layer_id = layer_id
        # Lớp chuẩn hóa norma
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    # Thiết lập phương thức forward phương thức này sẽ được gọi khi có các tham số truyền vào 
    # lớp 
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # Thực hiện tính toán attention
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        # chuyển kết quả của lớp attention và mạng thần kinh chuyển tiếp 
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# Xây dựng mô hình transformer 
class Transformer(nn.Module):
    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(self, params: ModelArgs):
        super().__init__()
        # từ điển tham số parameters 
        self.params = params 
        # kích thước của tập tình 
        self.vocab_size = params.vocab_size
        # number_layer 
        self.n_layers = params.n_layers 

        # Lớp tokenembedding sử dụng VocabParallelEmbedding để nhúng song song 
        # các luồng dữ liệu trên môi trường phân tán 
        self.tok_embedding =  VocabParallelEmbedding(
            # phương thức này nhận đầu vào là kích thước tập từ vựng 
            # dầu ra là danh sách mỗi token được biểu diễn bởi dim phần tử 
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        # Xây dựng một module List layer 
        self.layers = torch.nn.ModuleList()
        # Duyệt qua một vòng lặp 0-> num_layer -1 
        for layer_id in range(params.n_layers):
            # thêm từng lớp transformer vào danh sách modulist 
            self.layers.append(TransformerBlock(layer_id, params))

        # Định nghĩa một lớp chuẩn hóa 
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # phân tách tuyến tính cho tensor đầu vào theo cột kết quả nhận được các vector 
        # có cùng kích thước các vector này sẽ được chuyển cho các GPU trong môi trường phân 
        # tán để co thể thực hiện việc tính toán song song các vector 
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        # Định nghĩa tensor tần xuất tính toán tần suất của mỗi token trong chuỗi 
        # văn bản 
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    
    # @torch.inferance_model để chuyển mô hình sang chế độ suy luận 
    # với chế độ này các công việc khác sẽ bị ngừng thực thi như gradient .. 
    @torch.inferance_model()
    # phương thức forward sẽ được gọi khi các tham số được truyền vào 
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # lấy ra kích thước 
        _bsz, seqlen = tokens.shape
        # Thực hiện nhúng văn bản đầu vào 
        h = self.tok_embeddings(tokens)
        # học các tần suất của token 
        self.freqs_cis = self.freqs_cis.to(h.device)
        # Thực hiện cắt vector tần suất theo chỉ số start của token  
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        # khởi tạo mặt nạ mask = None
        mask = None
        # Xây dựng mặt nạ mask 
        # để xây dựng mặt nạ ta cần xác định seq_len 
        if seqlen > 1: 
            # xây dựng mask tensor shape (seqlen , seqlen) dtype = float-int 
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            # hàm triu được gọi để xây dựng mask chia mask thành ma trận tam giác các 
            # phần dưới trái sẽ = 1 
            mask = torch.triu(mask, diagonal=1)


            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output