from typing import Optional, Tuple 
import torch 
import torch.nn as nn 

class SiglipVisionConfig: 

    def __init__(
            self, hidden_size=768, 
            intermediate_size= 3072, 
            num_hidden_layers = 12, 
            num_attention_heads = 12, 
            num_channels = 3, 
            image_size= 224, patch_size = 16, 
            layer_norm_eps = 2e-6, 
            attention_dropout=0.0, 
            num_image_tokens: int = None, **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        # tham số  intermediate_size chỉ định kích thước của lớp giữa thuộc kiến 
        # trúc mạng FFn hoặc MLP 
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

    
# Xây dựng lớp nhúng hình ảnh dưới dạng các bản vá hình ảnh 
# có kích thước 16 * 16 và thêm thông tin về nhúng vị trí của các bản vá này 
class SiglipVisionEmbeddings(nn.Module):
    # Thiết lập phương thức khởi tạo
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Khởi tạo lớp nhúng hình ảnh sử dụng lớp mạng thần kinh tich chấp 
        # Conv2D để học các nhúng hình ảnh 
        self.patch_embedding = nn.Conv2d(
            # nhận đầu vào là 1 hình ảnh nguyên mẫu với số chiều biểu diễn là 3 
            in_channels=config.num_channels, 
            # đầu ra sẽ là các biều diễn nhúng có chiều = hidden_size
            out_channels= config.hidden_size,
            # sử dụng các bộ lọc kernel có kích thước = kích thước patches 
            kernel_size= self.patch_size , 
            # mỗi lần trượt sẽ trượt qua mỗi patch_size như vậy ta có thể lấy được 
            # các patch riêng lẻ không trùng nhau 
            stride = self.patch_size, 
            padding= "valid", # Với Valid sẽ không có phần đệm được thêm vào 
        )

        # Tính toán số lượng bản vá = kích thước ảnh 
        # chia hết cho kích thước lô ** 2 Vd nêu Image size = 49 * 49  , patch_size = 7
        # sẽ có 49 patch với kích thước 7 * 7 
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # mục tiêu là cần nhúng vị trí sao cho mỗi patches sẽ có 1 vị trí tương ứng 
        self.num_positions = self.num_patches 
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # Đăng ký tensor như một buffer trong mô hình có nghĩa là các tensor không phải là trọng số 
        # parameters của mô hình những vẫn được quản lý, tensor này sẽ không tham gia vào quá trình tối 
        # ưu hóa tham số 
        self.register_buffer(
            # Tên của tensor này 
            "position_ids",
            # khởi tạo tensor mở rộng tensor này thành tensor có kích thước 1 , num_position
            torch.arange(self.num_positions).expand((1, -1)),
            # tham sôs persitent chỉ định buffer này sẽ không được lưu trữ trong file trạng thái của mô hình 
            # khi gọi đến thuộc tính model.state_dict().
            persistent=False,
        )


    # Thiết lập phương thức forward phương thức này sẽ được gọi khi lớp được gọi 
    # với các tham số được truyền vào 
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Lấy ra các kích thước của hình ảnh đầy vào 
        # batch_size, channels, H, W 
        _, _, height, width = pixel_values.shape
        # Lấy ra các nhúng bản vá từ hình ảnh bằng việc sử dụng mạng Conv2d
        # đầu ra của lớp này sẽ là 1 tensor có dạng [batch_size, embed_dim , num_patches_h, num_patches_W]
        patch_embeds = self.patch_embedding(pixel_values)
        # reshape tensor đầu ra về dạng [Batch_size, embed_dim, num_patches]
        embeddings = patch_embeds.flatten(2) # làm phằng các chiều thứ 2 trở đi điều này sẽ hợp 
        # nhất 2 chiều cuối cùng thành 1 chiều có kích thước num_patches_h * num_patches_w
        # chuyển vị tensor này về dạng tiêu chuẩn trước khi thực hiện nhúng vị trí 
        # và chuẩn bị đầu vào cho kiến trúc máy biến áp 
        embeddings = embeddings.transpose(1, 2) # chuyển vị 2 chiều cuối cùng cho nhau 
        # Thêm nhúng vị trí cho mỗi patch hình ảnh 
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
    


# Xây dựng lớp SiglipAttention để tính toán attention cho các hình ảnh đầu vào 
# được tách thành các patches size 16 * 16
class SiglipAttention(nn.Module):
    "Multi-Head Attention from Attention is All You Need papper"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # lấy ra thông tin đầu vào 
        batch_size, seq_len, _ = hidden_states.size() # với seq_len = num_patches
        # Thực hiẹn nhúng vector q, k, v 
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # reshape các tensor và chuyển vị chúng để có được kích thước tiêu chuẩn cho việc tính toán Attention
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        
        # Kiểm tra kích thước của tensor attention phải đảm bảo tensor này có kích thước
        # batch_size, self.num_heads, seq_len, seq_len như kỳ vọng 
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            # nếu không thì đưa ra một cảnh báo kích thước 
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Multiply the attention weiights by the value states
        # là kết quả của phéo nhân tensor attention_weight [batch_size, num_heads, seq_len, seq_len] & [batch_Size, num_heads, seq_length, head_dim]
        # kết quả cuối cùng là  tensor shape [batch_size, num_heads , num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # kiểm tra kích thước của tensot attention output xem có đảm bảo được kích thước kỳ 
        # vọng hay khônng , nếu không ném ra một lỗi thông tin về kích thước tenssor 
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # reshaop tensor đầu ra về kích thước là tensor 3 chiều tiêu chuẩn 
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
        

# Xây dựng khối mạng Multi Linear Proceptron 
# có kiến trúc tương tự như mạng FFN được sử dụng để đảm bảo ngữ cảnh của 
# tensor biểu diễn cũng như tính tuyến tính của các tham số 
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        # Khởi tạo lớp tuyến tính Linear Projection 
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states
    

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 