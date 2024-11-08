import copy
import math
from math import sqrt
from functools import partial, wraps

from vector_quantize_pytorch import VectorQuantize as VQ

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torchvision

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange 

# constants

MList = nn.ModuleList

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# decorators

# xây dựng eval_decorator là một decorator nhận vào là một hàm fn và trả về một hàm 
# inner . HÀM INNER sẽ lưu trạng thái huấn luyện hiện tại của mô hình 
# (model.training) sau đó chuyển mô hình sang chế độ đánh giá 
def eval_decorator(fn):
    # Xây dựng một hàm inner nhận đầu vào là mô hình và các tham số bất kỳ 
    # được chuyển vào 
    def inner(model, *args, **kwargs):
        # lưu lại trạng thái huấn luyện của mô hình gán kết quả cho biến was_training 
        was_training = model.training 
        # chuyển mô hình sang chế độ xác thực 
        model.eval()
        # áp dụng hàm fn lên model.eval và kết quả gán cho biến out 
        out = fn(model, *args, **kwargs)
        # và đặt lại trạng thái huấn luyện của mô hình như ban đầu 
        model.train(was_training) 
        # mục đích của decorator anyf là để đảm bảo rằng mô hình sẽ được đánh giá mà không thay đổi 
        # trạng thái huấn luyện của nó 

        # và trả về out 
        return out 
    # trả về hàm inner 
    return inner 


# Xây dựng phương thức reomve_vgg cũng là một decorator 
# sử dụng @wraps từ module Functools để giữ metadata của hàm gốc hàm fn 
def remove_vgg(fn):
    @wraps(fn)
    # Xây dựng một hàm inner 
    def inner(self, *args, **kwargs):
        # kiểm tra xem đối tượng self có thuộc tính vgg hay không 
        # vgg là mạng tích chập vgg 
        has_vgg = hasattr(self, 'vgg')
        # nếu có tồn tại vgg 
        if has_vgg: 
            # định nghĩa một vgg để lưu trữ vgg 
            vgg = self.vgg 
            # sau đó xóa bỏ đi vgg khỏi self bằng hàm delattr 
            delattr(self, 'vgg')
        
        # thực thi hàm fn 
        out = fn(self, *args, **kwargs)

        # sau khi fn được thực thi nếu self ban đầu có thuộc tính vgg 
        # thì thuộc tính này sẽ được khôi phục 
        if has_vgg: 
            self.vgg = vgg 
        
        # trả về kết quả của hàm fn 
        return out 

    # và trả về hàm inner 
    return inner 




# Các hàm trợ giúp đối số từ khóa 
# Xây dựng phương thực pick_and_pop nhận đầu vào là một danh sách keys
# và một từ điển d 
def pick_and_pop(keys, d):
    # áp dụng hàm map lên một hàm lambda để lấy ra các key tương ứng với danh sách keys trong từ điển d 
    # và kết quả được lưu vào dnah sách values 
    values = list(map(lambda key: d.pop(key), keys))
    # trả về một từ điển gồm danh sách keys và values tương ứng 
    return dict(zip(keys, values))


# Xây dựng hàm group_dict_by_key để nhóm từ điển qua tham số keys
# nhận đầu vào là một hàm điều kiện cond và một từ điển d 
def group_dict_by_key(cond, d):
    # khởi tạo return_evla là một dnah sách, danh sách này lưu trữ 2 từ điển rỗng 
    # dict 1 , dict 2 
    return_val = [dict(), dict()]
    # duyệt qua các keys trong từ điển d 
    for key in d.keys():

        # dựa vào kết quả là một boolean của hàm điều kiện cond để xác định key đó 
        # có phù hợp với điều kiện hay không 
        match = bool(cond(key))
        # đảo ngược kết quả của match nếu match = True => False 
        # áp dụng hàm int sẽ cho được kết quả True nếu = 0 và False = 1
        ind = int (not match)
        # return_cal[ind] sẽ truy cập vào từ điển thứ nhất nếu ind là 0 
        # hoặc từ điển thứ 2 nếu ind = 1(False)
        # return _eval[ind][key] = d[key] sẽ gán giá trị của key từ từ điển d vào từ điển con thích 
        # hợp trong return_eval dựa trên giá trị của ind 
        return_val[ind][key] = d[key]

    # trả về 1 tuple chứa 2 từ điển con 
    return (*return_val,)


# xây dựng hàm string_begins_with nhận đầu vào là một chuỗi đầu vào 
# và một chuỗi prefix và kiểm tra xem nó có bắt đầu bằng chuỗi prifix hay không 
def string_begins_with(prefix, string_input):
    # Nếu như chuỗi string_input bắt đầu bằng chuỗi prefix s
    # return sẽ trả về True và ngược lại 
    return string_begins_with.startswith(prefix)


# xây dựng hàm group_by_key_prefix để phân loại các khóa trong từ điển d 
# dựa vào tiền tô prefix 
def group_by_key_prefix(prefix, d): 
    # sử dụng hàm group_dict_by_key  để  phân loại các khóa trong từ điển d 
    # dựa trên tiền tố prefix 
    # Đầu tiên, nó sử dụng hàm partial(string_begins_with, prefix) để tạo một hàm mới, chuyển tiếp prefix vào hàm string_begins_with.
    # Sau đó, nó gọi hàm group_dict_by_key với hàm điều kiện đã tạo và từ điển d.
    return group_dict_by_key(partial(string_begins_with, prefix), d)
    # Kết quả trả về là một tuple chứa hai từ điển con: từ điển chứa các khóa bắt đầu
    # bằng prefix và từ điển chứa các khóa không bắt đầu bằng prefix


# xây dưng hàm groupby_prefix_and_trim Kết quả trả về là một tuple chứa hai từ điển: kwargs_without_prefix và kwargs.
# Hàm này sử dụng lại hàm group_dict_by_key và thêm một bước xử lý sau đó.
def groupby_prefix_and_trim(prefix, d):
    # Đầu tiên, nó gọi hàm group_dict_by_key giống như ở trên để phân loại 
    # các khóa trong từ điển d dựa trên tiền tố prefix.
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    # Sau đó, nó tách ra hai từ điển con từ kết quả trả về: kwargs_with_prefix (chứa các khóa bắt đầu bằng prefix)
    # và kwargs (chứa các khóa không bắt đầu bằng prefix).
    # Tiếp theo, nó tạo một từ điển mới kwargs_without_prefix bằng cách loại bỏ
    # tiền tố prefix khỏi các khóa trong kwargs_with_prefix.
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    # Kết quả trả về là một tuple chứa hai từ điển: kwargs_without_prefix và kwargs.
    return kwargs_without_prefix, kwargs



# Xây dưng các hàm chức năng với việc sử lý tensor 

# Xây dựng hàm log trả về kết quả của log(tensor + epsilon )
def log(t, eps =1e-10):
    return torch.log(t + eps)

# Xây dựng một hàm để áp dụng hình phạt cho gradient 
def gradient_penalty(images, output, weight = 10):
    # lấy ra kích thước batch_size 
    batch_size = images.shape[0]
    # tính toán gradient với tensor output và images
    gradients = torch_grad(outputs = output, inputs = images,
                           # xâ dựng tensor grad_outputs shape = output.size (full 1) để
                           # tính toán gradient theo images 
                           grad_outputs = torch.ones(output.size(), device = images.device),
                           # tham số create_grad = True cho phép tạo đồ thị tính toán để tính gradient của gradient (để tính gradient penalty).
                           # retain_graph = True giữ lại đồ thị tính toán để tính gradient penalty.
                           create_graph = True, retain_graph = True, only_inputs = True)[0]
                           # only_inputs = True chỉ tính gradient theo images.
                           # [0] lấy gradient theo image

    # chuẩn hóa gradient 
    # sử dụng hàm rearrange để thay đổi kích thước của tensor gradient cho phù hợp với 
    # bước tính toan tiếp theo 
    gradients = rearrange(gradients, 'b ... -> b (...)')
    # sử dụng gradients.norm(2, dim = 1) tính chuẩn bình phương Euclidean của gradient theo chiều thứ 1 (tức theo batch)
    # sau đó sử dụng mean để tính toán trung bình, của bình phương khoảng cách 
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean() 
    # kết qủa là trả về giá trị của gradient penalty nhân với hệ số trọng số weight


# xây dựng hàm chuẩn hóa l2norm 
def l2norm(t):
    # áp dụng hàm chuẩn hóa normalize cho tensor t 
    return F.normalize(t, dim=-1)

# xây dựng hàm kích hoạt leakyRelu 
# hàm này là một biến thể của hàm ReLu 
def leaky_relu(p = 0.1):
    return nn.LeakyReLU(0.1)

# Xây dựng hàm stable_softmax để tính toán softmax một cách ổn đingj 
def stable_softmax(t, dim = -1 , alpha = 32 ** 2):
    # chuẩn hóa phép chia tensor t cho hệ số alpha 
    t = t / alpha 
    # sau đó trừ tensor t cho giá trị lớn nhất của tensor t theo chiều dim và 
    # đặt keepdim để giữ lại không gian tính toán 
    # sử dụng hàm detach để ngắn chặn việc lan chuyển gradient qua một phần của đồ thị tính 
    # toán 
    t = t - torch.amax(t, dim = dim, keepdim = True).detach()
    # cuối cùng áp dụng softmax cho tensor chuẩn hóa t và cơ số alpha theo dim
    return (t * alpha).softmax(dim = dim)


#  xây dựng hàm safe_div được sử dụng để thực hiện phép chia một cách an toàn, tránh lỗi chia cho số gần bằng 0.
def safe_div(numer, denom, eps = 1e-8):
    # trả về kết quả của 1 phép chia tensor thông thường 
    return numer / (denom + eps)


# Xây dựng các hàm loss Gan 
# hàm loss cho discriminator nhận đầu vào là ảnh thật và ảnh giả 
def hinge_discr_loss(fake, real):
    # trả về kết quả là loss của WAGAN 
    # loss 1 / N * max(0.1 + fake) + max (0.1 -real)
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

# Xây dựng hàm loss cho generator 
def hinge_gen_loss(fake):
    # return loss 1 / N * Fake 
    return -fake.mean()

# Xây dựng hàm bce_discr hàm mất mát nhị phân cho trình phân biệt đối sử 
# tính toán sự khác biệt giữa xác suất dự đoán của ảnh fake và ảnh thật  
def bce_discr_loss(fake, real):
    # return loss = 1/N * - Log (1 - sigmoid(fake)) - log(sigmoid(real))
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

# Xây dựng hàm mất mát nhị phân cho generator 
# để tính toán sai lệch một trung bình tiêu cực của nhật ký phân biệt 
# đối xử với các mẫu được tạo 
def bce_gen_loss(fake):
    # return - 1/N * Log(sigmoid(fake))
    return -log(torch.sigmoid(fake)).mean()

# xây dựng hàm grad_layer hàm này tính toan độ dốc của tổn thất nhất định đối với một 
# lớp cụ thể (cho lan truyền ngược)
def grad_layer_wrt_loss(loss, layer):
    # sử dụng torch_grad để tính toan độ dốc 
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach() # kết quả là bị tách ra nghĩa là gradient bị ngăn chặn trở 
    # lại các lớp ban đầu 


# Xây dựng hàm chuẩn hóa cho VQGAN 
class LayerNormaChan(nn.Module):
    # Thiết lập phương thức khởi tạo 
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        # định nghĩa các thuộc tính epsilon và gamma
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1)) 

    # Thiết lập một phương thức forward sẽ được gọi khi có các
    # tham số được chuyển vao 
    def forward(self, x):
        # khởi tạo tensor var không sử dụng bias trong quá trình tính toán 
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        # Tính trung bình của tensor x theo chiều thứ 2
        mean = torch.mean(x, dim = 1, keepdim = True)
        # Thưcj hiện chuẩn hóa cho x 
        return (x - mean) / (var + self.eps).sqrt() * self.gamma 
    


# Xây dựng trình phân biệt đối xử discriminator cho mạng Gan 
class Discriminator(nn.Module):
    # Thiết lập phương thức khởi tạo 
    def __init__(self, dims, channels = 3, groups = 16 , init_kernel_size =5):
        super().__init__()
        # định nghĩa một thuộc tính dim_pairs 
        # sẽ chứa 2 kích thước dim = [:-1] và dim [1:] một danh sách lưu trũ các kích 
        # thước 0 -> -1 và từ 1 -> n 
        dim_pairs = zip(dims[:-1], dims[1:])
        # và một danh sách các module được khởi tạo như là các lớp ẩn 
        # xây dụng lớp layer đầu tiên  
        self.layers = MList(
            [nn.Sequential(
                nn.Conv2d(channels, dims[0], init_kernel_size, padding = init_kernel_size // 2),
                leaky_relu())
            ])
        
        # duyệt qua dnah sách pairs để lấy ra các cặp kích thước i và i + 1 
        # theo đó để tạo ra các cặp lớp alyer đầu vào và đầu ra dựa vào các cặp anyf 
        for dim_in , dim_out in dim_pairs:
            # tại mỗi lần lặp thêm 1 lớp layers vào module List 
            # dưới dạng danh sách tuần tự 
            self.layers.append(nn.Sequential(
                # các lớp anyf được xây ựng bằng Conv2d với đầu vào và đầu ra tương ứng
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                # sử dụng hàm chuẩn hóa group để thực hiện nhóm dữ liệu thành các nhóm
                # và thực hiện chuẩn hóa giá trị trên mỗi nhóm riêng lẻ  
                nn.GroupNorm(groups, dim_out),
                # cuối cùng sau mỗi khối này áp dụng một hàm Leaky_Relu để biến đổi tuyến tính 
                leaky_relu()
            ))

        # tính toán kích thước dim 
        dim = dims[-1]
        # thuộc tính to_logits là một lớp tuần tự khác 
        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            # một lớp tích chập 1 * 1 để giảm số lượng kênh xuống còn dim 
            nn.Conv2d(dim, dim, 1),
            leaky_relu(), #hàm kích hoạt 
            nn.Conv2d(dim, 1, 4) # Một lớp tích chập 4x4 khác giảm số kênh xuống còn 1 (để phân loại nhị phân).
        )

    # Thiết lập phương thức forward để áp dụng tính toán tiến trình cho đầu vào x
    def forward(self, x):
        # duyệt qua dnah sách các lớp layer 
        for net in self.layers:
            # áp dụng các lớp anyf cho x 
            x = net(x)

        # trả về kết quả phân loại nhị phân 
        return self.to_logits(x)
    

# Xây dựng lớp nhúng position encoding cho hình ảnh 
class ContinuousPositionBias(nn.Module):
    """from https://arxiv.org/abs/2111.09883 with swin transformer v2"""

    # Thiêt lập phương thức khởi tạo và định nghia các thuộc tính 
    def __init__(self, *, dim, heads, layers =2):
        super().__init__()
        # Định nghĩa một mạng network dạng module List 
        # chưa được khởi tạo 
        self.net = MList([])
        # thêm vào mạng net một Lớp layer tuyến tính tuần tự 
        self.net.append(nn.Sequential(nn.Linear(in_features=2, out_features=dim), leaky_relu()))

        # sử dụng 1 vòng lặp từ 0 -> layers - 1 
        for _ in range (layers - 1):
            # thêm vào mạng net một lớp Linear tại mỗi lần lặp 
            self.net.append(nn.Sequential(
                    nn.Linear(dim, dim),
                    leaky_relu()))
            
        # sau đó thêm vào mạng net 1 lớp linear với đầu ra = Heads 
        self.net.append(nn.Linear(dim, heads))
        # và Đăng ký bộ nhớ tạm  self.register_buffer và đặt tên là 
        # 'rel_pos' mà không cần lưu trữ giá trị cố định trong quá trình lưu 
        # và tải mô hình do persistent = False 
        self.register_buffer('rel_pos', None, persistent = False)

    # Thiết lập phương thức forward sẽ được gọi khi nhận các tham số tryền vào 
    def forward(self,x):
        # lấy ra kích thước x theo chiều cuối cùng và tên thiết bị 
        n, device = x.shape[-1], x.device 
        # Tính kích thước của bản đồ đặc trừng features_map bằng cách lấy căn bậc 2 của n
        fmap_size = int(sqrt(n))

        # nếu như bộ nhớ tạm thời rel_pos chưa tồn tại 
        if not exists(self.rel_pos):
            # tạo một dãy số 0 -> fmap_size - 1 shape = fmap_size * fmap_size 
            pos = torch.arange(fmap_size, device= device)
            # Tạo ra một lưới vị trí chéo 2d bằng cách ghép các giá trị từ tensor pos 
            # shape = [fmap_size, fmap_size] sau đó sử dụng hàm stack để nối các lưới đã 
            # tạo thành tensor 3D shape [2, size, size]
            grid = torch.stack(torch.meshgrid(pos, pos, indexing= 'ij'))
            # Sử dụng hàm rearrange để sắp xếp lại lưới grid để đảm bảo 
            # mỗi vị trí trên lưới được biều diễn bằng mỗi vector 2 chiều 
            grid = rearrange(grid, 'c i j -> (i, j) c')
            # Tính vị trí tương đối cho mọi căp điểm trên lưới 
            # bằng cách lấy chỉ số hành của lưới thứ nhất - chỉ số cột của lưới thứ 2
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            # sử dụng hàm sign cho tensor rel_pos. Hàm này trả về một tensor mới mà mỗi 
            # phần tử có giá trị  là  -1, 0 hoặc 1 tùy vào dấu của phần tử tương ứng 
            # trong rel_pos. sỬ Dụng hàm log tự nhiên sau khi cộng thêm 1 cho trị tuyệt đối của
            # phần tử rl_pos để đảm bảo không có giá trị âm 
            rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            
            # Đăngn ký một real_poss như một buffer và không cần lưu trữ cố định 
            self.register_buffer('rel_pos', rel_pos, persistent = False)

        # chuyển đổi rel_pos sang kiểu float 
        rel_pos = self.rel_pos.float()

        # áp dụng từng lớp mạng lên rel_pos để học các bias vị trí 
        for layer in self.net:
            rel_pos = layer(rel_pos)

        # bias = rearrange(rel_pos, 'i j h -> h i j'): Sắp xếp lại rel_pos để có dạng phù hợp với tensor đầu vào x.
        bias = rearrange(rel_pos, 'i j h -> h i j')
        # return x + bias: Cộng bias vừa tính được vào tensor đầu vào x và trả về kết quả.
        return  x + bias 
    
        


# Xây dựng mạng Resetnet Encoder / Decoder 
class ResnetEncDec(nn.Module):
    # Thiết lập phương thức khởi tạo 
    def __init__(
        self,
        dim, channels = 3,
        layers = 4,layer_mults = None,
        num_resnet_blocks = 1,
        resnet_groups = 16, first_conv_kernel_size = 5,
        use_attn = True, attn_dim_head = 64,
        attn_heads = 8, attn_dropout = 0.,
    ):
        super().__init__()
        # đảm bảo rằng kích thước dim chia hết cho các nhóm của reset_net_groups 
        # tức là chia hết cho số lượng nhóm chuẩn hóa groups 
        assert dim % resnet_groups == 0, f'dimension {dim} must be divisible by {resnet_groups} (groups for the groupnorm)'

        # định nghĩa một thuộc tính layer 
        self.layers = layers 

        # Định nghĩa cho khối mã hóa và giải mã 
        self.encoders = MList([])
        self.decoders = MList([])

        # định nghĩa một thuộc tính multi_layers là một danh sách chứa các phần tử 
        # là bội số của phép nhân 2 ** t với t = 0 -> layers - 1 
        layer_mults = default(val=layer_mults, d= list(map(lambda t: 2 ** t, range(layers))))
        # đảm bảo rằng số phần tử của dannh sách layers bằng với số phần tử tromg layer_mult 
        assert len(layer_mults) == layers, 'layer multipliers must be equal to designated number of layers'
        
        # Khởi tạo một danh sách layer_dim sẽ là kích thước biểu diễn cho các lớp ẩn 
        # bằng dim ban đầu * với các kích thước trong layer_mults
        layer_dims = [dim * mult for mult in layer_mults]
        # tạo một tupel dim chứa kích thước ban đầu dim và các kích thước trong layer_dims
        # toán tử * được dùng để né danh sách layer_dims vào tuple 
        dims = (dim, *layer_dims)

        # khưởi tạo một interator dim_pairs chứa các cặp kích thước của dims 
        # theo nhau i -> j -> n 
        dim_pairs = zip(dims[:-1], dims[1:])

        # sử dụng hàm append để thêm phần tử t vào cuối dnah sách arr 
        # kết quả cuối cùng gán cho biến append 
        append = lambda arr, t: arr.append(t)
        
        # và  Hàm prepend thêm phần tử t vào đầu danh sách arr.
        prepend = lambda arr, t: arr.insert(0, t)


        # kiểm tra xem num_resnet_blocks có phải là một tuple hay không 
        if not isinstance(num_resnet_blocks, tuple):
            # ta nén các phần từ thành một tuple bằng cách nén các góa trị 0 lặp lại 
            # (layers - 1) lần sau đó thêm giá trị num_resnet_blocks vào cuối tuple 
            # với toán tử * đựoc sử dụng để nén đầu tiên nén một giá trị lặp *(layer -1)
            # và * nén các giá trị 0 lặp lại layers - 1 thành 1 tuple con và gộp với num_resnet_blocks 
            # để có được 1 tuple 
            num_resnet_blocks = (*((0,) * (layers - 1)), num_resnet_blocks)

        # tương tự như trên kiểm tra xem use_attn có phải là một tuple hay không 
        if not isinstance(use_attn, tuple):
            # Nếu không ta nén các giá trị False lặp lại (layers - 1) lần sau đó thêm 
            # giá trị use_attn vào cuối tuple 
            use_attn = (*((False,) * (layers - 1)), use_attn)

        
        # sử dụng một vòng lặp duyệt qua các danh sách và lấy ra phần tử của chúng 
        # hàm zip được sử dụng để kết hợp các chỉ ố lớp ... 
        for layer_index, (dim_in, dim_out), layer_num_resnet_blocks, layer_use_attn in zip(range(layers), dim_pairs, num_resnet_blocks, use_attn):
            # thêm các lớp layer tương ứng với số lần lặp vào encoder 
            # với kích thước dim được thay đổi sau mỗi lần lặp 
            append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1), leaky_relu()))
            # Tương tự như trên tại mỗi lần lặp thêm 1 lớp Tíc chập ngược 2D vào danh sách 
            # với cùng kích thước kernel , stride và padding sau đó áp dụng hàm kích hoạt Leaky_ReLU 
            prepend(self.decoders, nn.Sequential(nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1), leaky_relu()))

            # Nếu như tham số layers use_attention số lớp attention 
            if layer_use_attn:
                # thực hiện thêm một lớp attention VQGA vào đầu danh sách các lớp giải mã 
                # với các tham số cần thiết 
                prepend(self.decoders, VQGanAttention(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, dropout = attn_dropout))

            # Vòng lặp for duyệt qua 0-> n layers - 1 của khối resetnet
            for _ in range(layer_num_resnet_blocks):
                # Thêm các khối Reset Network Blocks vào danh sách các lớp encoder  
                # và các khối GLURESBLOCKS vào đầu danh sách các lớp decoder
                append(self.encoders, ResBlock(dim_out, groups = resnet_groups))
                prepend(self.decoders, GLUResBlock(dim_out, groups = resnet_groups))

            # Nếu như layer_use_attention = True 
            if layer_use_attn:
                # Ta thực hiện thêm một lớp Attention VQGANATTENTION vào danh sách các lớp 
                # encoder 
                append(self.encoders, VQGanAttention(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, dropout = attn_dropout))

        # prepend(self.encoders, nn.Conv2d(...)): Thêm lớp tích chập 2D đầu tiên vào đầu danh sách các lớp mã hóa, với số kênh đầu vào (channels), 
        # kích thước đặc trưng (dim), và kích thước kernel (first_conv_kernel_size).
        prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding = first_conv_kernel_size // 2))
        # append(self.decoders, nn.Conv2d(...)): Thêm lớp tích chập 2D cuối cùng vào cuối danh sách các lớp giải mã, 
        # chuyển từ kích thước đặc trưng (dim) trở lại số kênh đầu vào (channels).
        append(self.decoders, nn.Conv2d(dim, channels, 1))

    
    # Xây dựng phương thức get_encoded_featuremap_size 
    # để lấy ra kích thước của fearturé map sau khi được mã 
    # hóa qua tất acr các lớp mạng 
    def get_encoded_fmap_size(self, image_size):
        # giả định rằng sau khi qua mỗi lớp thì features trong không gian sẽ giảm 
        # đi 1/2 kích thước do bước nhảy strides = 2 
        # Vì vâyk kích thước cuối cùng sẽ được tính như sau Image_size // 2 ^ number_layers (kích thước này sẽ đạt giới hạn khi đến layer cuối)
        return image_size // (2 ** self.layers)
    

    # Xây dựng phương thức last_dec_layer
    # là một thuộc tính của lớp này được sử dụng để truy xuất trọng số của lớp 
    # Decoder cuối cùng 
    @property 
    def last_dec_layer(self):
        # trả về trọng số lớp decoder cuối cùng 
        return self.decoders[-1].weight 
    
    # Xây dựng phương thức encode để áp dụng các lớp encoder lên đầu vào x 
    def encode(self, x):
        # duyệt qua các lớp encoder áp dụng từng lớp encoder cho x 
        for enc in self.encoders: 
            x = enc(x)

        # và trả về kết quả cuối cùng của x 
        return x 


    # Xây dựng phương thức decoder đẻ áp dụng các lớp decoder lên x 
    # nhận x là đầu ra của khối encoder 
    def decoder(self , x):
        # 
        for dec in self.decoders:
            # áp dụng từng lướp cho  x 
            x = dec(x)

        # return  x
        return x 
    

# Xây dựng lớp phương thức GLU RESETNETWORK Block 
# là một mạng tích chập để thực hiện tính toán và chuẩn hóa các kết quả 
class GLUResBlock(nn.Module):
    # Thiết lập phương thức khởi tạo và định nghĩa 
    # các thuộc tính 
    def __init__(self, chan, groups = 16):
        super().__init__()
        # Xây dựng mnagj net với các lớp layer 
        self.net = nn.Sequential(
            # mạng tích chập 2D 
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            # Hàm kích hoạt Gelu 
            nn.GLU(dim = 1),
            # Lớp chuẩn hóa Group normalize thực hiện chuẩn hóa trên các nhóm 
            # dữ liệu 
            nn.GroupNorm(groups, chan),
            # Sau đó áp dụng thêm 1 lớp tích chập 2d 
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            # và 1 lớp Kích hoạt 
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            # cuối cùng là lớp tích chập cuối cùng 
            nn.Conv2d(chan, chan, 1)
        )

    # Phương thức forward sẽ được gọi khi x được truyền vào 
    def forward(self, x):
        # truyền x vào mạng net 
        return self.net(x) + x
    


# Xây dựng lớp phương thức Res Block là lớp mạng 
# Residual Network 
class ResBlock(nn.Module):
    # Thiết lập phương thức khởi tạo
    def __init__(self, chan, groups = 16):
        super().__init__()
        # định nghĩa khối mạng Residual 
        self.net = nn.Sequential(
            # sử dụng 1 lớp tích chập 
            nn.Conv2d(chan, chan, 3, padding = 1),
            # sau đó thực hiện chuẩn hóa nhóm trước khi áp dụng hàm kích hoạt 
            nn.GroupNorm(groups, chan),
            # sau khi chuẩn hóa áp dụng hàm kích hoạt leaky_relu 
            leaky_relu(),
            # Tương tự như trên 
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 1)
        )

    # Phương thức này sẽ được gọi khi  x được đưa vaò
    def forward(self, x):
        # trả về kết quả của khối Residual 
        return self.net(x) + x
    


# Xây dựng lớp phương thức VQGanAttention 
class VQGanAttention(nn.Module):
    #Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(self, *, dim, dim_head = 64, heads = 8, dropout = 0):
        super().__init__()
        # định nghĩa các thuộc tính
        self.heads = heads 
        self.scale = dim_head ** -0.5 # = 8 
        inner_dim = heads * dim_head # = 512 


        # định nghĩa một lớp Dropout layer 
        self.dropout = nn.Dropout(dropout)
        # định nghĩa một lớp chuẩn hóa phân phối dữ liệu 
        self.pre_norm = LayerNormaChan(dim)

        # Định nghĩa một lớp Relative Position Bias Table để tính toán 
        # bias vị trí tương đối 
        self.cpb = ContinuousPositionBias(dim = dim // 4, heads = heads)

        # Theo sau đó ta định nghĩa 2 lớp tích chập qkv và to_out 
        # với qkv để tính toán việc nhúng vector qkv 
        self.to_qkv = nn.Conv2d(dim,inner_dim * 3, 1, bias= True)
        # và to_out để tính toán đầu ra 
        self.to_out = nn.Conv2d(inner_dim, dim, 1,  bias = True)


    # Thiết lập phương thức forward để thực hiện xây dựng 
    # việc tính toán attention 
    def forward(self, x):
        # lấy ra số lượng heads đầu chú ý 
        h = self.heads 
        # lấy ra kích thước h w của hình ảnh và sử dụng x.clone 
        # tạo ra một bản sao độc lập của tensor x  
        height, width , residual = *x.shape[-2:], x.clone()

        # Thực hiện chuẩn hóa trước cho x
        x = self.pre_norm(x)

        # Tính toán các vector q, k, v dử dụng lớp Conv2d sau đó sử dụng hàm chunk 
        # để tách thành 3 tensor lẻ theo chiều thứ 2 
        # khi đó tensor shape [batch_size, 512, channels, w, h]
        q, k, v = self.to_qkv(v).chunk(3, dim = 1)

        # sau đoc chuyển vị các tensor này từ shape [batch_size, num_heads * head_dim (or channels_per_head), x, y]
        # thành tensor shape [ batch_size, num_heads, channels_per_head, w * h]
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = h), (q, k, v))

        # sử dụng việc tính toán Q * K.T sau đó nhân kết quả này với 
        # tỷ lệ scale khi đó tensor có dạng [batch_size, num_heads * head_dim, w * h, w * h] 
        sim = einsum('b h c i, b h c j -> b h i j', q , k) * self.scale 

        # sau đó áp dụng ma trận vị trí tương đối bias vào kết quả của Q.K.T 
        sim = self.cpb(sim)
        
        # áp dụng softmax cho kết quả này 
        attn = stable_softmax(sim, dim = -1)
        attn = self.dropout(attn)

        # sau đó nhân kết qảu attention với vector v để có được tensor 
        # result attention shape = [batch_size, hidden_dim(num_heads * head_dim), w*h , w*h] * [batch_size, num_heads, head_dim, w * h]
        out = einsum('b h i j, b h c j -> b h c i', attn, v)
        # sau đó chuyển vị tensor out thành dạng shape [batch_size, num_head * head_dim , w , h]
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x = height, y = width)
        out = self.to_out(out)

        # sau đó cộng tensor out với tensor x(residual) có cùng kích thước 
        return out + residual # để đảm bảo tính chât đầy đủ của dữ liệu 
    


# ViT encoder / decoder 

# Xây dựng lớp RearrangeImage để thực hiện việc thay đổi một kích thước 
# của tensor đầu vào 
class RearrangeImage(nn.Module):
    # XÂY DỰNG phương thức forward 
    # phương thức này sẽ được gọi khi có tham số truyền vào 
    def forward(self, x):
        # lấy ra kích thước chiều thứ 2 của x thường là số lượng pixel 
        n = x.shape[1]

        # tính căn bậc 2 của x để lấy ra w , h của ảnh 
        w = h = int(sqrt(n))
        #  sau đó là tái cấu trúc tensor 
        return rearrange(x, 'b (h w) ... -> b h w ...', h = h , w = w)
    

# Xây dựng lớp Attention để tính toán điểm chú ý attention 
class Attention(nn.Module):
    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(
        self, dim,
        *, heads = 8, dim_head = 32  ):
        super().__init__()
        # Định nghĩa một lớp LayerNorm chuẩn hóa dữ liệu
        self.norm = nn.LayerNorm(dim)
        # Định nghĩa thuôc tính head
        self.heads = heads
        # và tr lệ scale = căn bậc 2 của 32 
        self.scale = dim_head ** -0.5
        # và định nghĩa một inner dim = 256 
        inner_dim = dim_head * heads

        # Định nghĩa một thuộc tính to_qkv để tính toán các vector q, k, v
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # và định nghĩa một lớp tuyến tính đầu ra 
        self.to_out = nn.Linear(inner_dim, dim)


    # Thiết lập phương thức forward nó sẽ được gọi khi nhận tham số được 
    # truyền vào 
    def forward(self, x):
        # gán h = kích thước heads 
        h = self.heads

        # thực hiện chuẩn hóa trước x trước khi tính toán q , k, v vector 
        x = self.norm(x)

        # Tính toán vector q, k, v và tách chúng thành 3 tensor riêng lẻ 
        # shape [batch_size , num_patches, hidden_dim]
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # sau đó chuyển vị các tensor này để tái cấu trúc các vector q, k, v 
        # để có hình dnag phù hợp cho multi-head attention [batch_size, num_heads, num_patches, head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale  # Nhân q với tỷ lệ để điều chỉnh độ lớn của giá trị.

        # Tính toán điểm tương đồng giữa q và k. kết quả là 1 tensor có shape [batch_size, num_heads, num_patches, head_dim] * shape [batch_size, num_heads, num_patches, head_dim]
        # => shape [batch_size, num_head * head_dim , num_patches, mum_patches]
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # Trừ đi giá trị lớn nhất từ mỗi hàng để cải thiện tính ổn định số học.
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # Áp dụng softmax để chuyển sim thành các trọng số attention.
        attn = sim.softmax(dim = -1)

        # Tính toán đầu ra dựa trên các trọng số attention và v.
        # shape [batch_size , hidden_dim , num_patches , num_patches] * shape [batch_size, num_head, num_patches, head_dim]
        # => shape [batch_size, heads, num_patches, head_dims]
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Tái cấu trúc tensor đầu ra để phù hợp với hình dạng mong muốn.
        # shape = [batch_size, num_patches, num_head * head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Chuyển đổi tensor đầu ra cuối cùng thông qua to_out.
        return self.to_out(out)


    
         
# Thực hiện xây dựng mạng Feedforward 
def FeedForward(dim, mult = 4):
    # trả về một danh sách tuần tự các lớp layer
    return nn.Sequential(
        # gồm 1 lớp chuẩn hóa trước LayerNorm
        nn.LayerNorm(dim),
        # 1 lớp chiếu tuyến tính linear Projecting 
        nn.Linear(dim, dim * mult, bias = False),
        # Hàm kích hoạt Gelu
        nn.GELU(),
        # vÀ MỘT LỚP CHIẾU TUYẾN TÍNH CUỐI CÙNG 
        nn.Linear(dim * mult, dim, bias = False)
    )

# Xây dựng lớp Transformer 
class Transformer(nn.Module):
    # Thiết lập phuong thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(
        self,
        dim,
        *,
        layers,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        # Định nghĩa một thuộc tính layers là một Module List chưa được khởi tạo 
        self.layers = nn.ModuleList([])
        # duyệt qua danh sách các layers
        for _ in range(layers):
            # tại mỗi lần lặp thêm vào Module List 
            self.layers.append(nn.ModuleList([
                # một lớp Attention
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                # và một mạng Feedforward Network 
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        # cuối cùng định nghĩa một thuộc tính layernorm-
        self.norm = nn.LayerNorm(dim)

    # Thiết lập phương thức forward se được gọi khi có các tham số được truyền vào
    def forward(self, x):
        # duyệt qua danh sách cac lớp attn và ffn trong danh sách layers
        for attn, ff in self.layers:
            # áp dụng từng lớp attn và ffn cho tensor x 
            x = attn(x) + x
            x = ff(x) + x

        # cuối cùng trả về kết quả của x được chuẩn hóa 
        return self.norm(x)



# Xây dựng lớp VITEncoder & ViT Decoder 
class ViTEncDec(nn.Module):
    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(self, dim, channels=3, layers = 4, 
                patch_size = 8, dim_head = 32, heads = 8, ff_mult =4 ):
        super().__init__()
        # định nghĩa các thuộc tính 
        self.encoded_dim = dim 
        self.patch_size = patch_size

        # Định nghĩa một input_dim = 3 *  64
        input_dim = channels * (patch_size ** 2)

        # Định nghĩa một thuộc tính encoder là một danh sách tuần tự các layer 
        self.encoder = nn.Sequential(
            # Thêm một lớp mở rộng tensor đầu vào shape [batch_size, channels, (h * patches), (w* patches)]
            # thành một tensor shape [batch_size, (h * w), (patch_size * patch_size * channels)]
            # 
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # Tiếp theo thêm một lớp Chiếu tuyến tính Linear Projecting 
            nn.Linear(input_dim, dim),
            # Và khối Transformers với các tham số được truyền vào 
            Transformer(
                # dim kich kích thước của không gian biểu diễn 
                dim = dim,
                # kích thước biểu diễn của mỗi đầu chú ý 
                dim_head = dim_head,
                # số lượng đầu chú ý 
                heads = heads,
                # Kích thước của ff_dim
                ff_mult = ff_mult,
                # và số lượng lớp layers multihead Attention 
                layers = layers
            ),
            # sau đó thay đổi lại kích thước của hình ảnh 
            RearrangeImage(),
            # cuối cùng chuyển vị lại hình ảnh về kích thước ban đầu [batch_size, channels, h , w]
            Rearrange('b h w c -> b c h w')
        )

        # Định nghĩa một thuộc tính decoder 
        self.decoder = nn.Sequential(
            # Mở rộng lại tensor đầu vào shape [batch_size, [h *w], channels]
            Rearrange('b c h w -> b (h w) c'),
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers
            ),  # Lớp transformer trả về 1 tensor shape [batch_size, num_patches, hidden_dim]
            nn.Sequential(
                # xây dựng 1 lớp tuần tự khác tương tư như lớp fully connected 
                # trong mạng cnn 
                nn.Linear(dim, dim * 4, bias = False),
                # sử dụng tanh làm hàm kích hoạt 
                nn.Tanh(),
                nn.Linear(dim * 4, input_dim, bias = False),
            ),# sau đó mửo rộng và chuyển vị tensor biểu diễn 
            RearrangeImage(),
            # cuối cùng thực hiện tái cấu trúc lại tensor này về shape ban đầu 
            Rearrange('b h w (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )


    # Thiết lập phương thức get_encoded_feature_map_size để lấy ra kích 
    # thước của các features map 
    def get_encoded_fmap_size(self, image_size):
        # trả về kích thước của features map = image_size // self.patches 
        # vs nếu image_size = 64 patches_size = 8 => fmap = 8 * 8
        return image_size // self.patches 
    
    # Thiết lập một phương thức last_dec_layer l
    # được định nghĩa là một thuộc tính của lớp 
    @property
    def last_dec_layer(self):
        # trả về trọng số của lớp cuối cùng trong phần tử thứ 3 của lớp decoder 
        return self.decoder[-3][-1].weight
    
    # Thiết lập phương thức encode là lớp biểu diễn mã hóa của
    # Mô hình VIT 
    def encode(self, x):
        return self.encoder(x)

    # Tương tự thiết lập phương thức decoder là lớp biểu diễn
    # của mô hình VIT 
    def decode(self, x):
        return self.decoder(x)
    


# Xây dựng lớp NullVQGanVAE lớp này sẽ trả về 
# kết quả của phương thức khi nó được gọi 
# main vqgan-vae classes

class NullVQGanVAE(nn.Module):
    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính
    def __init__(
        self,
        *,
        channels
    ):
        super().__init__()
        # định nghĩa một thuộc tính encoded_dim 
        self.encoded_dim = channels
        #  thuộc tính layer 
        self.layers = 0

    # hàm get_encoed_fmao_size trả về size của f_map khi được truy xuất 
    def get_encoded_fmap_size(self, size):
        return size

    # hàm cpy_for_evla sẽ được sửu dụng để 
    # lấy tham số hoặc giá trị của tiến trình evaluation
    def copy_for_eval(self):
        return self

    # hàm encoded trả về kết quả của khối encoder
    def encode(self, x):
        return x

    # Tương tự hàm decoder trả về kết quả của hàm decoder 
    def decode(self, x):
        return x
    


# Xây dựng lớp Vector Quantization Generative Advesarial Network 
class VQGanVAE(nn.Module):
    # Thiết lập phương thức khởi tạo và định nghĩa các thuôc tính 
    #
    def __init__(self, *, dim, image_size, channels = 3, layers = 4, 
        # hàm chi phí l2 cho nhiệm vụ recontruction 
        l2_recon_loss = False, use_hinge_loss = True, vgg = None,
        # Kích thước của mỗi vector trong codebook = 256 , # Số lượng vector trong codebook = 512 
        vq_codebook_dim = 256, vq_codebook_size = 512, vq_decay = 0.8, vq_commitment_weight = 1.,   # Trọng số của commitment loss.
        # Khởi tạo codebook bằng K-means hay không.
        vq_kmeans_init = True, vq_use_cosine_sim = True, use_vgg_and_gan = True,
        vae_type = 'resnet', discr_layers = 4, **kwargs ):

        super().__init__()  # Gọi constructor của lớp cha nn.Module.
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)  # Tách các tham số có tiền tố 'vq_'.
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)  # Tách các tham số có tiền tố 'encdec_'.

        # Định nghĩa 3 thuộc tính image_size , channels , codebook_size 
        self.image_size = image_size 
        self.channels = channels 
        # và thuộc tính codebook_size thể hiện số luọng vector trong codebook = 512 
        self.codebook_size = vq_codebook_size 

        # kiểm tra xem kiến trúc VAE ĐƯỢC SỬ DỤNG LÀ GÌ 
        # NẾU LÀ RESET NETWORK 
        if vae_type == "resnet":
            # khởi tạo mô hình bằng kiến trúc Reset Net 
            enc_dec_klass = ResnetEncDec
        # Nếu là Vistion Transformer 
        elif vae_type == 'vit':
            # khởi tạo mô hình bằng kiến trúc ViTEncDec 
            enc_dec_klass = ViTEncDec
        # Còn lại nếu chuỗi vae_type khác 2 trường hợp trên 
        # Ném ra một lỗi không hợp lệ 
        else:
            raise ValueError(f'{vae_type} not valid')
        

        # Định nghĩa kiến trúc decode& encoder
        self.enc_dec = enc_dec_klass(
            dim = dim,
            channels = channels,
            layers = layers,
            **encdec_kwargs
        )

        # Định nghĩa mô hình Vector Quantization 
        self.vq = VQ(
            dim = self.enc_dec.encoded_dim,
            codebook_dim = vq_codebook_dim,
            codebook_size = vq_codebook_size,
            decay = vq_decay,
            commitment_weight = vq_commitment_weight,
            accept_image_fmap = True,
            kmeans_init = vq_kmeans_init,
            use_cosine_sim = vq_use_cosine_sim,
            **vq_kwargs
        )

        # Hàm chi phí tái toạn 
        # Nếu như điều kiện l2 = True sử dụng hàm mean_squared_error
        # còn không sử dụng hàm chi phí l1 
        self.recon_loss_fn = F.mse_loss if l2_recon_loss  else F.l1_loss


        # Tăt Gan và Mất tri giác nếu tỷ lệ GRAY tồn tain 

        # 1: Khởi tạo kiến trúc VGG Cnn 
        self.vgg = None 
        # 2: khởi tạo trình PHÂN BIỆT ĐỐI SỬ 
        self.discr = None 
        # 3: Khởi tạo thuộc tính use_vgg and gan 
        # cho phép sử dụng cả Gan và Vgg hoặc là không 
        self.use_vgg_and_gan = use_vgg_and_gan 

        # Nếu như tham số này không tồn tại lập tức trả về 
        if not use_vgg_and_gan: 
            return 
        
        # Hàm mất mát chi giác , Hàm này được sử dụng cho kiến trúc Discriminator 
        # của Gan được sử dụng để đánh gia mức độ chênh lệch giữa dữ liệu 
        # được dự đoán và dữ liệu thực tế. Và mục đích của nó nhằm cải thiện 
        # khả năng sinh dữ liệu giả mạo của mạng Generator sao cho chúng không thể phân biệt được với dữ liệu thật

        # nếu như vgg đã tồn tại 
        if exists(vgg):
            # định nghĩa thuộc tính VGG 
            self.vgg = vgg 
        
        # Nếu như kiến trúc này không có sẵn 
        else: 
            # định nghĩa mô hình VGG đã được đào tạo trước sử dụng kiến 
            # trúc vGG 16 lớp 
            self.vgg = torchvision.models.vgg16(pretrained=True)
            # sau đó xây dựng vgg phân loại bỏ qua 2 lớp layer cuối cùng 
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])


        # Xây dựng Gan và các hàm mất mát liên quan 
        # xây dựng một danh sách các kích thước mỗi phần tử trong danh sách là 
        # kết quả của phép lũy thừa 2 ** t [t in range (0 -> discr_layer)]
        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        # sau đó nhân các kích thước này với dim để có được danh sách mới 
        # gán kết qủa cho
        layer_dims = [dim * mult for mult in layer_mults]
        # sau đó sử dụng toán tử * để nén dnah sách kết quả là 1 tuple được gán cho dims
        dims = (dim, *layer_dims)

        # Định nghĩa Trình phân biệt đối sử của VQGan 
        self.discr = Discriminator(dims = dims, channels = channels)
        
        # Xây dựng các hàm chi phí cho discriminator và Generator
        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss


    # ĐỊNH NGHĨA PHƯƠNG THỨC ENCODED_DIM 
    # LÀ MỘT THUỘC TÍNH CỦA LỚP ĐƯỢC SỬ DỤNG ĐỂ TRUY XUẤT KÍCH THƯỚC DIM CỦA 
    # KHỐI ENCODER TRONG VQGanVAE 
    @property 
    def encoded_dim(self):
        # trả về kích thước dim của khối này 
        return self.enc_dec.encoded_dim
    
    # Xây dựng phương thức det_encoded_fmap_size được sử dụng để tính toán 
    # fearures map của ảnh sau khi xử lý mã hóa 
    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size=image_size)
    
    # Xây dựng phương thức copy_for_eval sử dụng phương thức deepcopy
    # để sao chép lại tham  
    def copy_for_eval(self):
        # lấy ra thiết bị đang được sử dụng 
        device = next(self.parameters()).device()
        # xây dụng hàm deepcopy 
        vae_copy = copy.deepcopy(self.cpu())

        # Nếu như điều kiện sau = True 
        if vae_copy.use_vgg_and_gan:
            # thì laoij bỏ đi tham số của discriminator 
            del vae_copy.discr
            # và tham số của vgg
            del vae_copy.vgg

        # sử dụng vae_copy để sao chép lại tham số ủa eval 
        vae_copy.eval()
        # và đồng thư=ời sao chép lại tên thiết bị 
        return vae_copy.to(device)
    

    # Xây dựng một phương thức decorator với remove_vgg 
    # sử dụng để loại bỏ hoặc thay đổi ác phần của trạng thái mô hình liên quan đến mô hình VGG,

    @remove_vgg
    # hàm state_dict 
    def state_dict(self, *args, **kwargs):
        # trả về một tù điển chứa toàn bộ trạng thái của mô hình, bao gồm các trọng 
        # số của ácc tham số 
        return super().state_dict(*args, **kwargs)

    # Tương tự như trên hàm laod_state_dict là một decorator 
    # sử dụng ddeer laoin bỏ hoặc thay đổi các phần trạng thái của mô hình liên qua 
    # đến vgg
    @remove_vgg
    # phương thức này tải lại trạng thái của mô hình từ một từ điển 
    # thươngd đựo sử dụng để khôi phục mô hình từ một checkpoint 
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    @property
    # @property là một decorator trong Python, được sử dụng để tạo ra 
    # một thuộc tính có thể truy cập như một attribute mà không cần phải gọi nó như một phương thức.
    def codebook(self):
        # Phương thức này trả về codebook từ mô hình vector quantization (VQ). 
        # Codebook là một tập hợp các vector nhúng được sử dụng để rời rạc hóa không gian nhúng của mô hình VAE.
        return self.vq.codebook

    # hàm encoder được sử dụng để giải mã các features map của hình ảnh 
    def encode(self, fmap):
        fmap = self.enc_dec.encode(fmap)
        # trả về biểu diễn nhúng của feature map 
        return fmap
    


    # XÂY DỰNG PHƯƠNG THỨC DECODE PHƯƠNG THỨC NÀY THỰC HIỆN 2 VIỆC 
    # Đầu tiên sử dụng mô hình vector quantization self.vq để rời rạc hóa 
    # feature_map thành accs indices trong codebook và tính toán commit_loss

    # sau đó sử dụng decoder của mô hình để giải mã fmap tời rạc hóa thành hình ảnh 
    # ban đầu hoặc mọt phiên bản tái tại hình ảnh 
    def decode(self, dmap, return_indices_and_loss = False):
        # sử dụng mô hình VQ để biến đổi fmap thành 
        # fmap , indices, và tính toán commit_loss 
        fmap, indices, commit_loss = self.vq(fmap)

        # sau đó giải mã fmap từ kết qủa của  mô hình vector quantization 
        # thành hình ảnh 
        fmap = self.enc_dec.decode(fmap)

        # kiểm tra xem có được chỉ định trả về danh sách chỉ số của ácc fmap 
        # và commit_loss của chúng không 
        if not return_indices_and_loss:
            # nếu không chỉ trả về ảnh fmap 
            return fmap

        # còn lại trả về đầy đủ các tham số 
        return fmap, indices, commit_loss
    

    # Thiết lập phương thức forward sẽ được gọi 
    # khi các tham số được truyền vào 
    def forward(
        self, img, return_loss = False,
        return_discr_loss = False, return_recons = False,
        add_gradient_penalty = True
    ):
        # lấy ra các kích thước của hình ảnh và tên thiết bị 
        batch, channels, height, width, device = *img.shape, img.device
        # đảm bảo rằng h và w  = image_size * image_size 
        assert height == self.image_size and width == self.image_size, 'height and width of input image must be equal to {self.image_size}'
        # và số lượng channels = kích thước channels được lấy ra 
        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

        # áp dụng lớp encode lên hình ảnh kết quả nhận được các biểu diễn fmap 
        fmap = self.encode(img)

        # sau đó sử dụng decode để giải mã các fmap thành hình ảnh fmap và chỉ số tương ưng 
        # cùng với commit_loss 
        fmap, indices, commit_loss = self.decode(fmap, return_indices_and_loss = True)

        # nếu 2 giá trị sau = False
        if not return_loss and not return_discr_loss:
            # chỉ lấy ra fmap 
            return fmap
        

        # đảm bảo rằng chỉ một trong hai loại loss được trả về: loss của autoencoder hoặc loss của discriminator.
        # Toán tử ^ là phép toán XOR, nghĩa là chỉ một trong hai biến return_loss hoặc return_discr_loss có thể là True.
        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'


        # whether to return discriminator loss
        # kiểm tra xem có cần trả về loss của discriminator 
        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            # sử dụng detach để ngăn chặn việc tính toán gradient cho fmap 
            # giúp tối ưu hóa hiệu suất khi không cần cập nhật gradient descent cho tensor này 
            fmap.detach_()
            # đảm bảo rằng gradient dược tính toán cho image
            img.requires_grad_()

            # áp dụng discriminator  lên cả fmap và image để lấy logits 
            # là đầu ra trước hàm softmax cho cả 2 tensor
            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))
            
            # Tính toán loss của discriminator dựa trên logits của fmap và img
            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)
            # Nếu cần thêm gradient penalty (một kỹ thuật để cải thiện ổn định của GAN), 
            # tính toán penalty và thêm vào loss của discriminator.
            if add_gradient_penalty:
                # Tính toán gradient pânlty cho image và img_discr_logits
                gp = gradient_penalty(img, img_discr_logits)
                # cộng loss của discriminator với hình phạt gradient 
                loss = discr_loss + gp

            # Nếu cần trả về hình ảnh được tái tạo (fmap), trả về cả loss và fmap.
            if return_recons:
                return loss, fmap

            # Nếu không cần trả về hình ảnh được tái tạo, chỉ trả về loss
            return loss

        # reconstruction loss
        # tính toán mất mát dựa trên hình ảnh được tái tạo lại
        recon_loss = self.recon_loss_fn(fmap, img)

        # trả về sớm nếu đào tạo trên thang độ xám 

        # kiểm tra xem use_vgg_and_gan có được thiết lập 
        if not self.use_vgg_and_gan:
            # nếu không kiểm tra thêm 1 điều kiện cho việc trả về ảnh được tái tạo
            if return_recons:
                # trả về loss của ảnh được tái tạo và ảnh được tái tạo
                return recon_loss, fmap

            # TRƯỜNG HỢP KHÔNG TRẢ VỀ ẢNH TA CHỈ NHẬN ĐƯỢ LOSS
            return recon_loss

        # HÀM MẤT MÁT CHI GIÁC 
        # CỦA ẢNH THỰC VÀ ẢNH TÁI TẠO 
        img_vgg_input = img
        fmap_vgg_input = fmap

        # Nêu như chiều thứ 2 của ảnh shape = 1 tức channels = 1 
        # có nghĩa là ta đang sử dụng nền xám 
        if img.shape[1] == 1:
            # handle grayscale for vgg
            # chuyển đổi nó sang RGB 
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

        # sau đó sử dụng mạng vgg để mã hóa đặc trưng cho image
        img_vgg_feats = self.vgg(img_vgg_input)
        # và các hình ảnh được tái tạo lại
        recon_vgg_feats = self.vgg(fmap_vgg_input)
        # Hàm mất mát chi giác sử dụng hàm mse để tính toán độ sai lệch 
        # của ảnh thực và ảnh tái tạo
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # Hàm mất mát cho Generator 
        # đựoc tính dự trên sai lệch sự phân biệt của ảnh được generator 
        # tạo ra bởi Discriminator 
        gen_loss = self.gen_loss(self.discr(fmap))

        # Tính toán trọng số phù hợp nục đích của việc này là để điều chỉnh trọng số 
        # của hàm mát mát chi giác dựa trên tỷ lệ gradient của nó 
        # so với gradient của loss tổng quát giúp cân bằng hai loại loss này trong quá trình huấn luyện.

        # Lấy lớp cuối cùng của decoder từ mô hình enc_dec để tính toán gradient.
        last_dec_layer = self.enc_dec.last_dec_layer

        # Tính chuẩn L2 (norm) của gradient của loss tổng quát (gen_loss) đối với trọng số của lớp cuối cùng (last_dec_layer)
        # Chuẩn L2 của một vector là căn bậc hai của tổng bình phương các phần tử của nó.
        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        # Tính chuẩn L2 của gradient của perceptual loss (perceptual_loss) đối với trọng số của lớp cuối cùng.
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        # Tính trọng số thích ứng bằng cách chia chuẩn của gradient đối với perceptual loss
        # cho chuẩn của gradient đối với loss tổng quát
        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        # Giới hạn giá trị của adaptive_weight để không vượt quá 10,000. Phương thức clamp_ 
        # sẽ thay đổi adaptive_weight trực tiếp mà không tạo ra một bản sao.
        adaptive_weight.clamp_(max = 1e4)


        # combine losses
        # hàm mất mát kết hợp 
        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        # kiểm tra kiểu trả về 
        if return_recons:
            return loss, fmap

        return loss