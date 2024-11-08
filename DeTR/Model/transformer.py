"""
Detr Transformer class. 

Copy-paste from torch.nn.Transformer with modification: 
    * positional encodings are in MHAttention. 
    * extra LN the end of encoder is removed 
    * decoder returns a stack of actvations from all decoding layers

"""

import copy 
import typing
from typing import Optional , List 
import torch 
import torch.nn.functional as F 
from torch import nn, Tensor 

# Xây dựng lớp phương thức Transformer dựa trên kiến trúc của mô hình Transformer 
class Transformer(nn.Module):
    # Thiết lập phương thức khởi tạo và cấu hình các tham số 
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
            num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
            activation='relu', normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        # định nghĩa lớp Transformer encoder layer 
        # với các tham số được truyền vaò d_model(embedding dim) , num_heads, ff_dim , dropout , ac, normalize
        encoder_layer = TransformerEncoderlayer(d_model, nhead, dim_feedforward, dropout, 
                                    activation, normalize_before)
        # Lớp Layernorm với tham số d_model và 1 điều kiện để lớp này tồn tại giằng 
        # nếu như Normalize_before = True 
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None 
        # Lớp Transformer Encoder lớp này nhận các tham số gồm encoder_layer
        # num_layers là số lớp và đầu ra của encoder_norm 
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Lớp Transformer Decoder Layer 
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, 
                                        dropout , activation, normalize_before)
        
        # Lớp decoder layerNormalize cho bộ phận decoder
        decoder_norm = nn.LayerNorm(d_model)
        # Lớp xử lý Transformer Decoder
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # một thuộc tính lớp reset_parameters để đặt lại các tham số 
        self._reset_parameters()

        # định nghĩa thuộc tính d_model và n_head 
        self.d_model = d_model
        self.nhead = nhead

    
    # Thiết lập phương thức _reset_parameters
    def _reset_parameters(self):
        # duyệt qua danh sách các tensor trong từ điển parameters và gán nó cho p 
        for p in self.parameters():
            # kiểm tra chiều của tensor đảm bảo nó > 1 
            if p.dim() > 1: 
                # nêu như các tensor p (hay mỗi một tensor p thỏa mãn) thì áp
                # dụng phân phối đồng nhất xavier (để thực hiện phân phối đồng nhất khởi tạo lại tham số trong mạng giúp 
                # cải thiện tốc độ học tập và hiệu suất)
                nn.init.xavier_uniform_(p)

    # Thiết lập phương thức Forward (chuyển tiếp) thực hiện xây dựng xử lý mô hình 
    # Transformer  nhận đầu vào gômg src ma trận đầu vào , mặt nạ mask , ma trận nhúng truy vấn 
    # và ma trận position embedding 
    def forward(self, src, mask , query_embed, pos_embed):
        # làm phẳng ma trận src có shape = N*C*H*W [BACH_SIZE , CHANNELS, HEIGHT, WIDTH]
        # thành ma trận shape = [HW *N , *C]
        # 1: lấy ra các chiều của tensor src 
        bs, c , h , w = src.shape 
        # làm phẳng tensor theo chiều 2 sau đó chuyển vị nó 
        # shape = N, C , H*W -> H*W, N , C
        src = src.flatten(2).permute(2, 0, 1)
        # áp dụng tương tự với tensor pos 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # kết quả tensor pos_embed sẽ có shape tuonwg tư như src
        # Thêm một chiều mới và ma trận 2 chiều query với unsqueeze và lặp lại tensor đó theo chiều thứ 2
        # kết quả là tensor mới 3 chiều shape [1, bs, 1]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # làm phẳng ma trận mask theo chiều 1 trở đi 
        mask = mask.flatten(1)

        # Xây dựng lên một tensor = 0 shape = shape [query_embed] gán nó cho tgt 
        tgt = torch.zeros_like(query_embed)
        # Thực hiện áp dụng Transformer encoder cho đầu vào thực hiênh nhiệm vụ 
        # giải mã các hình ảnh 
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # sau đó chuyển tiếp kết quả cho Transformer Decoder 
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # sau đó chuyển vị trí 2 chiều của tensor hs cho nhau 
        # shape = [N , D, H*W] -> shape = [N, H*W , D] và áp dụng chuyển vị cho tensor encoder 
        # sau đó reshape = [N, c, h, w] (N có thể là batch_size or num_patches)
        return hs.transpose(1,  2), memory.permute(1, 2, 0).view(bs, c, h, w)




# Xây dựng hàm get_clones để tạo N bảo sao độc lập của Module Pytorch và trả về 
# dạng mộ ModuleList 
# nhận đầu vào gồm module là nguồn và N là số bản sao 
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    # thiết lập phương thức khởi tạo và cấu hình các tham số 
    def __init__(self, encoder_layer, num_layers, norm=None):
        # cho phép kế thừa các thuộc tính lớp 
        super().__init__()
        # định nghĩa thuộc tính layer sao chép lại 1 dnah sách các layer encoder 
        # với số lớp N = num_layer
        self.layers = _get_clones(encoder_layer, num_layers)
        # định nghĩa tham số thuộc tính num_layers
        self.num_layers = num_layers
        # và norm
        self.norm = norm
    
    # Xây dựng phương thức forward sẽ được gọi khi chạy mô hình 
    def forward(self, src ,
            # với 1 lựa chọn mask tensor , pos tensor và src_key_padding_mask
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None    ):
        # gán lại cho output = src
        output = src 
        # duyệt qua danh sahcs các layer và áp dụng các layer encoder cho dữ liệu 
        for layer in self.layers:
            # áp dụng lần lượt các layers lên đầu vào, truyền vào các tham số như mặt nạ , đệm mặt nạ ..
            # và tensor position 
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        # kiểm tra xem norm co = None
        if self.norm is not None:
            # nếu không = None thực hiện chuẩn hóa tensor output và thực hiện phân nói 
            # các giá trị trong tensor 
            output = self.norm(output)
        # trả về kết quả 
        return output






class TransformerDecoder(nn.Module):
    # Thiết lập phương thức khởi tạo lớp và cấu hình các tham số thuộc tính 
    def __init__(self, decoder_layer, num_layers,
             norm=None, return_intermediate=False):
        super().__init__()
        # định nghĩa tham số thuộc tính 
        # 1 : sao chép các decoder layer với n lớp gán cho thuộc tính layer 
        self.layers = _get_clones(decoder_layer, num_layers)
        # 2 : layernorma 
        self.norm = norm 
        # 3 : return_intermediate là tham số kiểm soát xem mô hình có trả về các lớp trung gian hay klhoong 
        # nếu = False mô hình sẽ trả về đầu ra từ lớp decoder cuối 
        # nếu = True tất cả các lớp decoder sẽ trả về 
        self.return_intermediate = return_intermediate

    # Tương tự như lớp phương thức Encoder xây dựng 1 phương thức forward 
    # sẽ được gọi khi gọi lớp này 
    def forward(self, tgt, memory,
                # nhận đầu vào là các tensor tham số 
                # và định nghĩa các lựa chọn tensor khởi tạo = None
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, # một lựa chọn là memory_mask tham số này 
                # cho phép học từ mặt nạ trước cho việc cải thiện các huấn luyện sau 
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                # tuwong tự là lựa chọn vs pos tensor và query_pos tensor
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # gán ouput = tgt (là tensor đầu vào 1)
        output = tgt

        # khởi tạo 1 danh sách intermediate để lưu trữ các kết quả 
        intermediate = []
        # duyệt qua một danh sách các layers gán nó cho layer
        for layer in self.layers: 
            # áp dụng từng layer lên tensor đầu vào tgt và memory 
            output = layer(output, memory, tgt_mask=tgt_mask,
                           # truyền vào các tham số cần thiết 
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           # cuối cùng là pos và query_pos 
                           pos=pos, query_pos=query_pos)
            
            # tại mỗi lần lặp được kết thúc thì kiểm tra xem tham số intermediate = True không 
            if self.return_intermediate:
                # nếu có thì ta áp dụng lần lượt norm cho mỗi output và thêm nó vào dnah sách lưu trữ
                intermediate.append(self.norm(output))

        # nếu như norm != None có nghĩa là nó tồn tại 
        if self.norm is not None : 
            # áp dụng layernorm cho danh sách output 
            output = self.norm(output)
            # kiểm tra xem intermediate = True 
            if self.return_intermediate: 
                # xóa danh sách intermediate = pop 
                intermediate.pop()
                # sau đó thêm output vào tronh danh sách 
                intermediate.append(output)

        # nếu intermediate = True 
        if self.return_intermediate:
            # nối các output dưới dạng stack 
            return torch.stack(intermediate)

        # trả về outpt và thêm chiều vào vị trí 0
        return output.unsqueeze(0)
        


# xây dựng lớp phương thức transformer encoder thực hiện tính toán 
# chú ý và xây dụng bộ phận ffn cho attention encoder 
class TransformerEncoderlayer(nn.Module):
    # Thiêt lập phương thức khởi tạo 
    # nhận đầu vào là bộ các tham số cho mô hình cần xử lý 
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                activation='relu', normalize_before=False):
        super().__init__()
        # 1 : Định nghĩa một thuộc tính attention 
        # là lớp MultiHead Attention 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # 2 : Linear projection thực hiện công việc chiếu tuyến tính dữ liệu 
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # 3 : Dropout layer bỏ đi những noron mà có ngưỡng đầu ra <= 0.5 
        self.dropout = nn.Dropout(dropout)
        # 5 : Lớp chiếu tuyến tính dữ liệu thứ 2 
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 6 : 2 lớp layer normalization 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 7 : áp dụng lấy 2 lớp dropout từ (3)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 8 : Định nghĩa một lớp Activations funcion 
        self.activation = _get_activation_fn(activation)
        # và 9 : thuộc tính normalize_before 
        self.normalize_before = normalize_before

        # Để biết thêm chi tiết về các lớp xem lại Mô hình Transformer 2017 

    
    # xây dựng phương thức nhúng vị trí cho tensor 
    def with_pos_embed(self, tensor , 
            # một lựa chọn pos tensor
            pos: Optional[Tensor]
            ):
        # trả về tensor nếu pos = none còn không cộng tensor với tensor pos 
        # kết quả là tensor biểu diễn postion_embedding 
        return tensor if pos is None else tensor + pos 
    
    # Thiết lập phương thức forward pos sẽ được gọi tùy thuộc vào tham số normalize_before 
    # thực hiện việc tính toán Attention 
    def forward_post(self,
                     src,
                    # tryền vào các tensor nguồn và lựa chọn tensor 
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # thực hiện tính toán vector Q , k bằng phương thức with_pos_embed 
        q = k = self.with_pos_embed(src, pos)
        # Tính toán Attention cho Q , K , V 
        # với các tham số như mask , padding_mask 
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0] # [0] có nghĩa là chỉ lấy dnah sách kết quả bỏ qua weight cuar attention 
                            # thông thưường kết quả của attention sẽ trả về 2 giá trị là score attention và weight attention 
        # áp dụng dropout cho attention 
        src = src + self.dropout1(src2)
        # sau đó thực hiện chuẩn hóa và phân phối dữ liệu qua lớp layernorma
        src = self.norm1(src)
        # áp dụng phép chiếu tuyến tính lên dữ liệu hay (ffn layer)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # dropout linear 
        src = src + self.dropout2(src2)
        # thực hiện công việc chuẩn hóa cuối cùng 
        src = self.norm2(src)
        # trả về kết quả src cuối cùng 
        return src
    

    # Thiết lập phương thức forward pre sẽ được gọi tùy thuộc vào tham số normalize_before 
    # thực hiện việc tính toán Attention . Phương thức này tương tự như trên nhưng ta áp dụng chuẩn 
    # hóa dữ liệu trước khi thực hiện nhúng 
    def forward_pre(self,
                     src,
                    # tryền vào các tensor nguồn và lựa chọn tensor 
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        
        # áp dụng chuẩn hóa phân phối dữ liệu trước khi thực hiện nhúng 
        src2 = self.norm1(src)
        # thực hiện tính toán vector Q , k bằng phương thức with_pos_embed 
        q = k = self.with_pos_embed(src, pos)
        # Tính toán Attention cho Q , K , V 
        # với các tham số như mask , padding_mask 
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # áp dụng dropout cho attention 
        src = src + self.dropout1(src2)
        # sau đó thực hiện chuẩn hóa và phân phối dữ liệu qua lớp layernorma
        src = self.norm1(src)
        # áp dụng phép chiếu tuyến tính lên dữ liệu hay (ffn layer)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # dropout linear 
        src = src + self.dropout2(src2)
        # thực hiện công việc chuẩn hóa cuối cùng 
        src = self.norm2(src)
        # trả về kết quả src cuối cùng 
        return src

    # Xây dựng phưuowng thức forward phuuwong thức này sẽ được gọi khi các tham số được chuyển 
    # vào cho mô hình 
    def forward(self, src, 
                # các tensor lựa chọn 
                src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None, 
                pos: Optional[Tensor] = None
                ):
        # kiểm tra xen tham số Normalize_before = True 
        if self.normalize_before: 
            # nếu nó có tồn tại áp dụng tính toán Attention có chuẩn hóa trước 
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # còn không ta áp dụng tính toán Attention không chuẩn hóa 
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)




class TransformerDecoderLayer(nn.Module):
    # thiết lập phương thức khởi tạo và cấu hình các tham số thuộc tính 
    def __init__(self,  d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # định nghĩa các tham số thuộc tính cho lớp 
        # số lượng các lớp , bộ phận sử lý được gọi theo đúng kiến trúc ban đầu 
        # của Transformer 2017 của Google AI 
        # 1 : 2 bộ phận multihead Attention 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # 2: 1 lớp Linear projection sử dụng để chiếu tuyến tính cho dữ liệu
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # 3: Dropout attention layer or dropout ffn layer 
        self.dropout = nn.Dropout(dropout)
        # 4: lớp Linear Projection thứ 2 
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 5: Xây dựng 3 lớp layernormalization Layer cho việc thực hiện phân phối 
        # và chuẩn hóa dữ liệu 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # 6 : 3 lớp dropout layer từ lớp dropout (3)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 7 : Định nghĩa cho một hàm kích hoạt 
        self.activation = _get_activation_fn(activation)
        # 8 : Tham số normalize_before 
        self.normalize_before = normalize_before

    # xây dựng phương thức nhúng vị trí cho tensor 
    def with_pos_embed(self, tensor , 
            # một lựa chọn pos tensor
            pos: Optional[Tensor]
            ):
        # trả về tensor nếu pos = none còn không cộng tensor với tensor pos 
        # kết quả là tensor biểu diễn postion_embedding 
        return tensor if pos is None else tensor + pos 

    # Thiết lập phương thức forward_post để thực hiện tính toán Attention và thực 
    # hiện chuẩn hóa lớp 
    def forward_post(self, tgt , memory, tgt_mask: Optional[Tensor] = None,
                    # các lựa chọn tensor 
                    memory_mask: Optional[Tensor]  = None, 
                    tgt_key_padding_mask: Optional[Tensor] = None, 
                    memory_key_padding_mask: Optional[Tensor] = None,
                    # và 2 lựa chọn tensor pos và query_pos 
                    pos: Optional[Tensor] = None, 
                    query_pos: Optional[Tensor] = None
                    ):
        # tính toán vector query , key cho nhiệm vụ attention bằng with_pos_embed 
        q = k = self.with_pos_embed(tgt, query_pos)
        # tính toán attention cho q , k , v kêu gọi đến self.attn 
        # và chuyền vào các tham số cần thiết mask , padding _mask 
        tgt2 = self.self_attn(q, k , value=tgt, attn_mask=tgt_mask, 
                        key_padding_mask=tgt_key_padding_mask)[0] # lấy kết quả của Attention
        # áp dụng 1 lớp dropout cho kết quả Attention 
        tgt = tgt + self.dropout1(tgt2)
        # Thực hiện chuẩn hóa lớp và phân phối dữ liệu cho tgt 
        tgt = self.norm2(tgt)
        # Tính toán multihead Attention 
        tgt2 = self.multihead_atten(query=self.with_pos_embed(tgt, query_pos),
                        key=self.with_pos_embed(memory, pos),
                        value=memory, attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask)[0] # chỉ lấy kết quả của multihead 
        # thực hiện dropout multihead-Attention
        tgt = tgt + self.dropout2(tgt2)
        # thực hiện chuẩn hóa dữ liệu 
        tgt = self.norm2(tgt)
        # Thực hiện 1 lớp chiếu tuyến tính ánh xạ dữ liệu 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # áp dụng một lớp rời bỏ và chuẩn hóa cuối cùng 
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # trả về tensor biểu diễn của các lớp decoder 
        return tgt
    

    # Thiết lập phương thức forward_pre để thực hiện tính toán Attention và thực 
    # hiện chuẩn hóa lớp . Phương thức này áp dụng chuẩn hóa dữ liệu đầu vào trước khi 
    # thực hiện nhúng vị trí
    def forward_pre(self, tgt , memory, tgt_mask: Optional[Tensor] = None,
                    # các lựa chọn tensor 
                    memory_mask: Optional[Tensor]  = None, 
                    tgt_key_padding_mask: Optional[Tensor] = None, 
                    memory_key_padding_mask: Optional[Tensor] = None,
                    # và 2 lựa chọn tensor pos và query_pos 
                    pos: Optional[Tensor] = None, 
                    query_pos: Optional[Tensor] = None
                    ):
        # áp dụng chuẩn hóa trước 
        tgt2 = self.norm1(tgt)
        # tính toán vector query , key cho nhiệm vụ attention bằng with_pos_embed 
        q = k = self.with_pos_embed(tgt, query_pos)
        # tính toán attention cho q , k , v kêu gọi đến self.attn 
        # và chuyền vào các tham số cần thiết mask , padding _mask 
        tgt2 = self.self_attn(q, k , value=tgt, attn_mask=tgt_mask, 
                        key_padding_mask=tgt_key_padding_mask)[0] # lấy kết quả của Attention
        # áp dụng 1 lớp dropout cho kết quả Attention 
        tgt = tgt + self.dropout1(tgt2)
        # Thực hiện chuẩn hóa lớp và phân phối dữ liệu cho tgt 
        tgt = self.norm2(tgt)
        # Tính toán multihead Attention 
        tgt2 = self.multihead_atten(query=self.with_pos_embed(tgt, query_pos),
                        key=self.with_pos_embed(memory, pos),
                        value=memory, attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask)[0] # chỉ lấy kết quả của multihead 
        # thực hiện dropout multihead-Attention
        tgt = tgt + self.dropout2(tgt2)
        # thực hiện chuẩn hóa dữ liệu 
        tgt = self.norm2(tgt)
        # Thực hiện 1 lớp chiếu tuyến tính ánh xạ dữ liệu 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # áp dụng một lớp rời bỏ và chuẩn hóa cuối cùng 
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # trả về tensor biểu diễn của các lớp decoder 
        return tgt
    

    # Xây dựng phương thức forward phuuwong thức này sẽ được gọi khi các tham số được truyền vào lớp 
    def forward(self, tgt, memory,
                # nhận đầu vào là các tensor biểu diễn 
                # và khởi tạo danh sách các lựa chọn tensor 
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # kiểm tra điều kiện xem giá trị normalize_before có = True 
        if self.normalize_before:
            # nếu có áp dụng phép chuẩn hóa trước nhúng vị trí cho việc tính toán Chú ý 
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        # còn lại không áp dụng chuẩn hóa 
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)






# Xây dựng mô hình Transformer đầy đủ 
def build_transformer(args):
    # trả về mô hình TRansformer 
    return Transformer(
        # truyên vào nó các tham số càn thiết 
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        # các tham số này được trích xuất từ từ điển tham số của mô hình 
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        # và return_intermediate = True cho mục đích lấy các bộ tham số decoder
        return_intermediate_dec=True,
    )



# Thiết lập phương thức xây dựng các hàm kích hoạt 
def _get_activation_fn(activation):
    """"Return an activation function given a string"""
    # nếu như hàm kích hoạt = relu 
    if activation == "relu":
        # trả về hàm relu từ torch.nn
        return F.relu
    # tuuwong tự nếu là gelu 
    if activation == "gelu":
        return F.gelu
    # và nếu là glu 
    if activation == "glu":
        return F.glu
    
    # còn không ném ra một cảnh báo lỗi 
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")