import torch 
from torch import nn 
import torch.nn.functional as F 
from einops import rearrange 

# Xây dựng hàm crossEntropyLoss 
class LaMDA_Loss(nn.Module):
    # thiết lập phương thức khởi tạo 
    def __init__(self):
        super(LaMDA_Loss, self).__init__()

    # Thiết lập phuuwong thức forward phương thức này sẽ được gọi khi 
    # có các tham số truyền vào 
    def forward(self, x_input, x_labels):
        # lấy ra x_input và x_labels từ 2 danh sách đầu vào 
        # 1: Lấy ra một danh sách x_intputs là các giá trị được biểu diễn trong chiều cuối cùng 
        # của danh sách thứ nhất cắt danh sách này 1 khoảng từ 0-> embed_dim -1 
        # 2: danh sách các giá trị biểu diễn được lấy từ chỉ số 1 -> embed_dim 
        x_inp , x_labels = x_input[:, :-1], x_labels[: 1:]
        # tính toán chi phí chéo giữa 2 danh sách 
        loss = F.cross_entropy(rearrange(x_inp, "b c n -> b n c"), x_labels)
        return loss



# autoregressive wrapper

#xây dựng một hàm log 
def log(t, eps=1e-9):
    return torch.log(t + eps)

# Xây dựng một hàm top_k để tính toán các xác xuất cao nhất 
def top_k(logits, three=0.9):
    # tính toán số lượng k 
    k = int((1 - three) * logits.shape[-1])
    # tính toán topk kết quả nhận được một danh sách chứa giá trị và một danh sách 
    # chứa các chỉ số theo val 
    val, ind = torch.topk(logits, k)
    # lấp đầy một tensor shape = logits là các chỉ số int được làm gần lên 
    probs = torch.full_like(logits, float('-inf'))
    # từ danh sách logits ta lấy ra các giá trị values theo chỉ số ind 
    # với hàm scatter_ thay thế các chỉ số idx thành các giá trị tương ứng với nó 
    probs.scatter_(1, ind, val)
    # trả về tensor probs chứa giá trị được dự đoán 
    return probs


# Thiết lập lớp phương thức AUTOREGRESSIVEWRAPPER được sử dụng để tạo ra các
# chuỗi dự đoán tự động từ mô học
class AutoregressiveWrapper(nn.Module):
    # thiết lập phương thức khởi tạo và định nghĩa các tham số thuộc tính
    def __init__(self, net, max_seq_len = 512, pad_value = 0):
        super().__init__()      
        # pad_value là thuộc tính biểu thị cho phần đệm   
        self.pad_value = pad_value
        # net là mô hình mạng noron
        self.net = net
        # thuộc tính max_sequence_length 
        self.max_seq_len = max_seq_len

    @torch.no_grad() # sử dụng để bỏ qua việc tính toán gradient cho phương thức 
    # generate 
    def generate(
        self, 
        start_tokens, 
        seq_len, 
        eos_token = None, # token kết thúc 
        temperature = 1.0, # Tham số điều chỉnh độ đa dạng của dự đoán 
        filter_logits_fn = top_k, # hàm top_k để lọc các giá trị logits 
        filter_thres = 0.9, # một ngưỡng để lọc cá giá tị logits 
        **kwargs
        ):
        # lưu lại trạng thái huấn luyện của mô hình 
        was_training = self.net.training 
        # lấy ra hình dạng của danh sách tart_token là một danh sách đầu vào 
        _, t = start_tokens.shape 

        # đặt mô hình vào chế độ đánh giá trong chế độ này accs lơp như drop , batch_norm 
        # sẽ không được kích hoạt 
        self.net.eval()
        # tạo một danh sahcs out là chuỗi dự đoán = start_tokens 
        out = start_tokens
        
        # duyệt qua một khoảng 0 ->seq_length 
        for _ in range(seq_len):
            # lấy ra tokens là một danh sách từ chỉ số  n- self.max_seq đến chỉ số n cuối cùng 
            x = out[:, -self.max_seq_len:]
            # tính toán logits cho token tiếp theo bằng cách gọi mô hình 
            # với lát cắt [:, -1, :] tức là lấy phần tử cuối cùng theo chiều thứ 2
            # thì chiều thứ 2 này là x đang cần dự đoán nếu x cần dự đoán có idx = 10 thì danh sách này 
            # hiện tại có chiều thứ 2 = 10 
            logits = self.net(x, **kwargs)[:, -1, :]
            # Lọc logits bằng hàm filter_logits_fn (ví dụ: top-k filtering).
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)

            # Áp dụng nhiễu Gumbel và lấy ra token có xác suất cao nhấ
            gumbel_noise = -log(-log(torch.zeros_like(filtered_logits).uniform_(0, 1)))
            sample = ((filtered_logits / temperature) + gumbel_noise).argmax(dim=-1)

            # nối token này vào chuỗi dự đoán out 
            out = torch.cat((out, sample[:, None]), dim=-1)

            # kiểm tra nếu có token kết thúc (eos_token) và tất cả các token dự đoán đều 
            # là token kết thúc, thì dừng lại 
            if eos_token is not None and (sample == eos_token).all():
                break

        # thực hiện loại bỏ các token khởi đầu, ban đầu khỏi chuỗi dự đoán 
        out = out[:, t:]
        # Khôi phục trạng thái huấn luyện ban đầu của mô hình.
        self.net.train(was_training)
        #Trả về chuỗi dự đoán out.

    # thiết lập phương thức forward 
    def forward(self, x, **kwargs):
        # trả về mang các tham số của net theo x 
        return self.net(x, **kwargs)

