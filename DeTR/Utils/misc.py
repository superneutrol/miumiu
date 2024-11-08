import os 
import subprocess 
import time 
# Gọi ra 2 phương thức là defaultdict và deque 
# Dèaultdict là một lớp phụ của dictionary thông thường . 
# nó cung cấp một giá trị mặc dịnh cho các khóa không tồn tại 
# khi truy cập vào khóa không có trong dictionary defaultdict sẽ tự dộng tạo ra 
# khóa đó với giá trị mặc định mà được cung câp khi khởi tạo . giúp tránh lỗi keyerror

# Với deque là 1 danh sách được tối ưu hóa cho việc thêm và xóa các phần tử 
# từ cả 2 đầu . Deque hữu ích khi cần sử lý một hàng đợi queue hoặc ngăn xếp 
# stack với hiệu suất cao 
from collections import defaultdict, deque 
import datetime 
import pickle 
from packaging import version 
from typing import Optional , List 

import torch 
# phương thức torch.distributed trong pytorch cung cấp một gói giao tiếp 
# phân tán cho phép thực hiện huấn luyện song song trên nhiều máy tính . 
# nó hỗ trợ các hoạt động như > Gửi và nhận dữ liệu từ các quá trình (send/recv).
# Phát sóng một tensor từ 1 quá trình đến tất cả các quá trình khác . 
# Thu thập dữ liệu từ tất cả các quá trình và gộp chúng lại 
# Giamr dữ liệu  từ tất acr các quá trình theo một phép toán nhất định (reduce)
# Rào cản để đồng bộ hóa tât cả các quá trình (barrier)..
import torch.distributed as dist 
from torch import Tensor 

import torchvision 
# Kiểm tra phiên bản hiện tại của torch 
if version.parse(torchvision.__version__) < version.parse('0.7'):
    # nếu điều kiện đúng gọi thư viện torchvision.ops và torchvision.ops.misc 
    # để lấy ra phương thức _new_emty_tensor ( để tạo ra 1 tensor mới không chứa bất kỳ
    # dữ liệu nào emty tensor). Sử dụng để tạo tensor mà không cần giá trị cụ thể. 
    # và _output_size : được sử dụng để tính toán kích thước cho đầu ra dự kiến của 1
    # tensor sau khi áp dụng các phép biến đổi nhất định. 
    from torchvision.ops import _new_empty_tensor 
    from torchvision.ops.misc import _output_size 


# Thiết lập phương thức kiểm tra môi trường phân tán , trả về 1 giá trị True or False 
def is_dist_avail_and_initialized():
    # kiểm tra xem dist.is_avialable tức môi trường phân tán trên nhiều thiết 
    # bị đã có sẵn hay chưa 
    if not dist.is_available():
        # nếu chưa return False
        return False
    # tương tự kiểm tra xem môi trường phân tán đa thiết bị đã được khởi tạo hay chưa
    if not dist.is_initialized():
        # Nếu chưa return fale 
        return False
    # còn nếu tất cả đã tồn tại return True 
    return True


# Thiết lập phương thức SmoothedValue để làm mịn các luồng dữ liệu 
class SmoothedValue(object):
    """Track a series of value and provide accrs to smoothed values over window
        or the gllobal series average.
        Theo dõi 1 loạt các gái trị và cung cấp quyền truy cập để làm mịn được các giá trị 
        thông qua một cửa sổ hoặc mức trung bình của chuỗi toàn cầu .
    """
    # Thiết lập phương thức khởi tạo 
    def __init__(self, window_size=20, fmt=None):
        # Kiểm tra xem gía trị của biến fmt có tồn tại hay không 
        if fmt is None: 
            # gán fmt = 1 chuỗi định dnagj mặcd dịnh 
            # {median: 4.f} định dạng giá trị trịcuar median với 4 số sau dấu phẩy 
            # {global_avg:.4f}: Định dạng giá trị của global_avg (trung bình toàn cục)
            # với 4 chữ số sau dấu phẩy. Khi muốn hiển thị hoặc ghi lại các giá trị này , chuỗi 
            # định dạng sẽ được sử dụng để tạo ra một chuỗi có giá trị median và global_qvg
            # được định dạng theo cách đã chỉ định . 
            fmt = "{median:.4f} ({global_avg:.4f})"
        # gán biến deque  = 1 deque để có thể tối ưu được các chức năng xử lý đầu cuối của nó 
        self.deque = deque(maxlen=window_size)
        # Vaò 2 biến total và count để tính toán trung bình toàn cục
        # của tất cả ác giá trị đã thêm vào
        self.total = 0.0 
        self.count = 0
        # gán lại giá trị cho fmt = chuỗi có định dạng được khởi tạo như trên hoặc 
        # bằng với gía tri mặc định 
        self.fmt = fmt 
    
    # Thiết lập phương thức update để cập nhật các tham số đựoc sử dụng 
    # cho việc tính toán trung bình toàn cục của các giá trị
    def update(self, value, n=1):
        # Thêm vào danh sachs deque 1 giá trị value
        self.deque.append(value)
        # cộng giá trị count với n có nghĩa là cập nhật số lượng các giá trị 
        # đã thêm vào deque 
        self.count += n 
        # cộng giá trị total cho n * giá trị đầu vào value có nghĩa là 
        # cập nhật tổng giá trị tính cả trường hợp giá trị value được thêm vào 
        # nhiều lần = n* value 
        self.total += n * value

    # Thiết lập phương thức synchronize_between_processes để thực hiện việc đồng 
    # bộ hóa count và total giữa các quá trình trong môi trường phân tán 
    def synchronize_between_processes(self):
        """Waring: does not synchronize the deque!
            Cảnh báo không đồng bộ danh sách deque 
        """
        # kiểm tra xem môi trường phân tán có sẵn sàng và đã được khởi tạo hay không 
        if not is_dist_avail_and_initialized():
            # nếu có return 
            return 
        # tạo 1 tensor torch chứa các giá trị count và total và được phân bổ sử lý cho cuda
        t = torch.tensor(data=[self.count, self.total], dtype=torch.float64, device='cuda')
        # sử dụng hàm barrier() để tạo một dào cản đồng bộ hóa tất cả các quá trình 
        # cho đến điểm này 
        dist.barrier()
        # thực hiện một phép toán giảm tham số trênn tất cả các quá trình (t)
        # và kết hợp dữ liệu từ tất cả các quá trình t 
        dist.all_reduce(t)
        # Chuyển mỗi tensor t thành 1 danh sách để có thể cập nhật count và total 
        self.count = int(t[0]) # vì count có định dạng int 
        self.total = t[1] # còn total có định dạng float 

    # Thiết lập các phương thức thuộc tính giúp việc truy cập vào các phương thức này 
    # như là một thuộc tính thông thường không cần phải gọi chúng như các phương thức
    @property 
    # phương thức thuộc tính median 
    def median(self):
        # biến đổi danh sách deque thành 1 list sau đó ta biến đổi nó thành 1 tensor 
        d = torch.tensor(list(self.deque))
        # Trả về trung bình của các giá trị trong tensor d 
        return d.median().item()
         
    # tương tự như trên xây dựng phương thức avg đặt nó là 
    # một thuộc tính 
    @property 
    def avg(self):
        # tạo một tensor từ danh sách deque datatype = float 32
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        # tương tự như trên trả về trung bình các giá trị trong tensor d 
        return d.mean().item()
    # tương tự như trên tạo 1 thuộc tính là phương thức max 
    @property
    def max(self):
        # trả về giá trị lớn nhất của deque
        return max(self.deque)

    @property
    def value(self):
        # trả về giá trị cuối cùng của dnah sách deque 
        return self.deque[-1]

    # thiết lâpj phuuwong thức __str__ có chức năng chuyển 1 đối tượng thành chuỗi 
    def __str__(self):
        # trả về 1 giá trị theo định dạng của self.fmt với các 
        # giá trị thuộc tính đã được tính toán 
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
    

# thiết lập các phương chức chức năng 
    
# 1: get_word_size trả về số lượng các thiết bị hay thành phần tài nguyên tham 
# gia vào môi trường phân tán 
def get_world_size():
    # kiểm tra xem môi trường phân tán có sẵn sàng và 
    # nó đã được khởi tạo hay không is distributioned avialabled and initialized 
    if not is_dist_avail_and_initialized():
        return 1 
    # nếu co thì trả về kích thước của thế giới tính toán 
    return dist.get_world_size()


# 2 : get_rank trả về ID của quá trình  hiên tại trong môi trường phân tán
def get_rank():
    # tương tự như phuwong thức trên kiểm tra xem môi trường phân tán có đang sẵn ssangf 
    # và đã được khởi taọ hay chưa 
    if not is_dist_avail_and_initialized():
        # nếu không trả về id = 0
        return 0 
    # nếu có trả về ID của quá trình 
    return dist.get_rank()


# 3 : phương thức is_main_process kiểm tra xem quá trình hiện tại có phải là 
# quá trình chính hay không . Quá trình chính rank = 0
def is_main_process():
    return get_rank() == 0  # Kiểm tra xem quá trình hiện tại có phải
    #là quá trình chính (rank 0) hay không


# 4 : phương thức để lưu chữ  dữ liệu của quá trình chính. Nó sẽ gọi torch.save
# với các đối số được cung câps 
def save_on_master(*args, **kwargs):
    if is_main_process():
        # lưu trữ lại với các tham số tự động 
        torch.save(*args, **kwargs)


# Thiết lập phương thức all_gather để tập trung tất cả dữ liệu của các 
# quá trình trong cùng một môi trường phân tán 
def all_gather(data):

    """Run all_gather or arbitrary pickable data (no neccessarily tensors)
    Args:
        data : any pickable object
    Returns: 
        List[data]: list of data gathered from each rank.
    
    Chạy all_gather trên dữ liệu có thể chọn tùy ý (không nhất thiết phải là tensor)
    Trả về 1 danh sách dữ liệu được thu thập từ mỗi nhóm
    """
    # lấy ra kích thước word_size là số lượng quá trình tham gia vào môi 
    # trường tính toán phân tán . (có thể hiểu là số lượng thiết bị xử lý hay là số lượng GPU)
    word_size = get_world_size()
    # kiểm tra xem kích thước này  = 1 
    if word_size == 1:
        # trả về ngay danh scahs đầu vào 
        return [data]
    
    # Thực hiện tuần tự hóa 1 tensor 
    # dữ liệu data được chuyển đổi tuần tự (serialize) thành một chuỗi byte 
    # sử dụng thư viện pickable 
    buffer = pickle.dump(data)
    # sau đó ta biến đổi chuỗi byte (buffer) thành 1 ByteStorage (kho lưu trữ byte)
    storage = torch.ByteStorage.from_buffer(buffer)
    # sau đó chuyển kho lưu trữ này cho cuda
    tensor = torch.ByteStorage(storage).to('cuda')

    # Lấy ra kích thước của mỗi tensor cho cấp bậc 
    # Tạo một tensor chứa kích thước của tensor dữ liệu trên thiết bị CUDA
    # hàm tensor.numel = num_element số lượng phần tử trong tensor
    local_size = torch.tensor([tensor.numel()], device='cuda')
    # tạo một dnah sách các tensor để lưu trữ tất cả các tensor trên cùng 1 quá trình 
    # mỗi tensor trong danh sách chứa 1 giá trị = 0
    size_list = [torch.tensor([0], device='cuda') for _ in range(word_size)]
    # gọi đến hàm dist.all_gather để có thể tập trung tất cả các giá trị 
    # là các tensor trong cùng 1 quá trình và kích thước riêng lẻ của các tensor đó 
    dist.all_gather(size_list, local_size)
    # lặp qua 1 danh sách các tensor trên 1 tiến trình gán cho size 
    # sau đó lấy ra số lượng item trên mỗi tensor và gán nó = 1 giá trị int 
    # kết quả là 1 danh sách chứa từng kích thước của mỗi tensor trên 1 tiến trình trong 
    # môi trường phân tán 
    size_list = [int(size.item()) for size in size_list]
    # lấy ra kích thước lớn nhất tronh dnah sách 
    max_size = max(size_list)

    # Nhận tensor từ mọi cấp bậc receicing Tensor from all ranks 
    # đệm các tensor vì torch all_gather không hỗ trợ 
    # tập hợp lại tất cả các tensor có hình dạng khác nhau 
    tensor_list = []
    # duyệt qua dnah sách các tensor trong dnah sách size_list (một dnah sách chứa tất cả 
    # các tensor trên cùng 1 tiến trình) 
    for _ in size_list: 
        # Thêm vào dnah sách tensor_list một tensor trống có kích thước max_size 
        # là kích thước của tensor lớn nhất của tiến trình 
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    
    # kiểm tra xem kích thước của tensor local_size có bằng max_size 
    # điều này là một yêu cầu quan trọng trong việc đệm các tensor ngắn 
    if local_size != max_size: 
        # tạo một tensor đệm pad size = kích thước bù 
        # và kiểu byte uint8 để đồng bộ dữ liệu với tensor gốc 
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device='cuda')
        # Thực hiện đêmh tensor bằng cách nối các tensor này theo chiều thứ nhất 
        tensor = torch.concat((tensor, padding), dim=0)

    # Sử dụng dist.all_gather để thêm các tensor đã được đêm vào dnah sách tensor_list
    dist.all_gather(tensor_list,tensor)

    # ạo một danh scahs data_list = dỗng để lưu trữ dữ liệu đã được giải mã 
    data_list = []
    # sử dụng zip để phân giải các giá trị từ danh sách tensor_list và size_list
    # lấy ra kích thước và tensor gía trị tương ứng 
    for size , tensor in zip(size_list, tensor_list):
        # tensor.cpu().numpy(): Chuyển tensor từ thiết bị CUDA (GPU) 
        # về CPU và sau đó chuyển thành một mảng numpy.
        # .tobytes()[:size]: Chuyển mảng numpy thành một chuỗi byte và 
        # cắt chuỗi này theo kích thước thực tế của dữ liệu
        buffer = tensor.cpu().numpy().tobytes()[:size]
        #  Giải mã chuỗi byte trở lại thành dữ liệu gốc của nó.
        # và thêm nó vào danh sách data_list 
        data_list.append(pickle.loads(buffer))

    return data_list 

    
# Thiết lập phuuwong thức reduce_dict nhận đầu vào là các tệp từ điển 
# chứa tất cả ác tensor từ các quá trình 
# có chức năng chia nhỏ bớt các giá trị của mỗi tham số nhằm để 
# tất cả các quá trình có kết quả trung bình hoặc tổng , tùy thuộc vào 
# giá trị của tham số average 
def reduce_dict(input_dict , average=True):
    """Args: 
            input_dict (dict) : all the value will be reduce : tất cả các giá trị sẽ được chia 
            average (bool) : whether to do average or sum : Có nên tính trung bình hay không . 
        Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    # lấy ra kích thước của thế giới phân tán (thường là số lượng thiết bị sử lý trong môi trường phân tán)
    world_size = get_world_size()
    # kiểm tra xem world < 2
    if world_size < 2 : 
        # trả về ngay từ điển đầu vào 
        return input_dict
    
    # khi mà không có sử dụng trình tối ưu hóa gradient
    with world_size < 2 : 
        # tạo 1 danh scahs name để lưu trữ tên 
        names = []
        # và 1 dnah sách value để lưu trữ giá trị 
        values = []
        # duyệt qua dnah sách mà tên của các khóa trong từ điển đã được sắp xếp 
        for k in sorted(input_dict.keys()):
            # Lấy ra tên của nó tức k thêm vào dnah scahs name 
            names.append(k)
            # ánh xạ khóa k cho từ điển để lấy các giá trị của k thêm vào value
            values.append(input_dict[k])
        
        # sử dụng hàm torch.stack để nối danh sách các gía trị values theo k 
        # dạng ngăn xếp 
        values = torch.stack(values)
        # giảm tất acr các giá trị trên tất cả các tiến trình trong môi trường phân tán 
        dist.all_reduce(values)
        # kiểm tra xem average = true không 
        if average:
            # nếu có ta chia danh sách giá trị của tất cả ác tiến trình 
            # cho kích thước thế giới phân tán (số lượng các thiết bị tham gia vào mô truoiwngf)
            values /= world_size
        # xây dựng một từ điển mới gồm key và values là các giá trị của tất cả các tiến trình trong 
        # môi trường phân tán 
        reduced_dict = {k: v for k, v in zip(names, values)}
    # trả về từ điển . 
    return reduced_dict

# Xây dựng lớp trình ghi số liệu 
# và theo các chỉ số hoặc số liệu thống kế trong quá trình huấn luyện 
# hoặc đánh giá mô hình . 
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        # sử dụng defaultdict để định nghĩa một từ điển mặc định để lưu trữ các 
        # giá trị mặc định của đối tượng SmoothedValue 
        self.meters = defaultdict(SmoothedValue)
        # Delimiter được sử dụng để phân cách các giá trị khi in ra
        self.delimiter = delimiter

    # thiết lập phương thức để cập nhật các tham số cho từ điển meters 
    # lưu ý từ điển mà MetricLogger sử dụng mỗi key chỉ có 1 giá trị 
    def update(self, **kwargs):
        # Cập nhật các giá trị cho từng chỉ số trong từ điển
        # duyệt qua các cặp key và value trong từ điển (id và values)
        for k, v in kwargs.items():
            # Nếu giá trị là một tensor, chuyển nó thành một số Python
            if isinstance(v, torch.Tensor):
                # gán v = giá trị item đó 
                v = v.item()
            # Đảm bảo rằng giá trị là một số float hoặc int
            assert isinstance(v, (float, int))
            # Cập nhật giá trị cho chỉ số tương ứng
            self.meters[k].update(v)

    # Thiết lập phương thức xử lý __getattr__ là loại phương thức đặc biệt 
    # được gọi tự động khi cố gắng truy cập một thuộc tính không tồn tại của 
    # đối tượng.
    def __getattr__(self, attr):
        # phuuwong thức này được gọi khi thuộc tính attr không tìm thấy
        # trong từ điển meters 
        # nếu attr tồn tại trong từ điển meters 
        if attr in self.meters:
            # trả về giá trị của nó trong từ điển này 
            return self.meters[attr]
        # nếu attr tồn tại trong '__dict__' của đối tượng , trả về giá trịc ủa nó 
        #__dict__ chứa tất cả các thuộc tính của đối tương
        if attr in self.__dict__:
            # trả về giá trịc của nó trong đối tượng 
            return self.__dict__(attr)
        # nếu attr không tồn tại ném ra  ngoại lệ AttributeError.
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    # Tương tự như trên xây dựng 1 phương thức đặc biệt __str__
    # trả về một chuỗi biểu diễn đối tượng khi gọi str() or print()
    def __str__(self):
        # khởi tạo một danh sách loss_str 
        loss_str = []
        # duyệt qua từ điển meters lấy ra các key và value
        for name , meter in self.meters.items():
            # thêm chuổi biểu diễn của mỗi meter vào danh sách 
            loss_str.append("{}: {}".format(name, str(meter)))
        # Nối tất cả các chuỗi trong danh sách 'loss_str' bằng 'self.delimiter' và trả về.
        return self.delimiter.join(loss_str)
    
    # thiết lập phương thức synchronizer .. để thực hiện đồng bộ hóa 
    # các giá trịc ủa các meter giữa các quá trình 
    # điều này hữu ích trong môi trường tính toán phân tán 
    # đảm bảo rằng tất cả các quá trình có thông tin cập nhật nhất quán 
    def synchronize_between_processes(self):
        # duyệt qua các giá trị của từ điển meters gán cho meter 
        for meter in self.meters.values():
            # sau đó thực hiện đồng bộ hóa các giá trị cho mỗi quá  trình
            meter.synchronize_between_processes()

    # thiết lập phương thức add_meter vào từ điển meter theo khoa là name 
    def add_meter(self, name, meter):
        self.meters[name] = meter

    # Thiết lập phương thức log_every được sử dụng để ghi lại thông tin và in ra 
    # cosole tại các khoảng thời gian nhất định trong quá trình lặp qua 1 iterable 
    # (có thể là mọt danh sách hoặc một tập dữ liệu)
    
    # Hàm này sẽ được gọi trong 1 vòng lặp, và nó sẽ in ra thông tin theo định kỳ. 
    # dựa trên giá trị của print_freg. Thông tin được in ra bao gồm tiêu đề (nếu có), số 
    # lần lặp lại hiện tại, thời gian ước tính còn lại (eta), các chỉ số đang được theo dỗi 
    # thời gian lặp lại trung bình, và bộ nhớ tối đa sử dụng (nếu CUDA khả dụng)
        
    # Đoạn mã này giúp người dùng theo dõi tiến trình của quá trình lặp và đánh giá 
    # hiệu suất của mô hình hoặc thuật toán đang được chạy. 
    def log_every(self, iterable, print_freq, header=None):
        # khưởi tạo một biến đếm i = 0
        i = 0 
        # kiểm tra xem có tiêu đề hay không 
        if not header: 
            # gán cho head = 1 chuỗi rỗng 
            header = ''
            # ghi lại thời gian bắt đầu 
            start_time = time.time()
            # ghi lại thời gian kết thúc giá trị này sẽ được cập nhật sau mỗi lần lặp 
            end = time.time()
            # Tạo một đối tượng SmoothedValue để theo dõi thời gian lặp 
            iter_time = SmoothedValue(fmt='{avg:.4f}')
            # vào một đối tượng SmoothedValue để theo dõi thời gian dữ liệu 
            data_time = SmoothedValue(fmt='{avg:.4f}')
            # định dạng chuỗi cho số lượng chử số 
            # interable là 1 đối tượng có thể lặp len(iterable) lấy ra số lượng (interable)
            # sau đó str(len(iterable)) để biến giá trị số đó = 1 chuỗi 
            # và sau đó len(str(len(iterable))) lấy ra độ dài chuỗi đó 
            space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
            # kiểm tra xem cude có đang khả dụng hay không 
            if torch.cuda.is_available():
                # định dạng cho số hiện tại và tổng số lần lặp 
                # tức là tạo một chuỗi định dạng để in thông tin log trong quá trình lặp 
                # qua iterable. 
                log_msg = self.delimiter.join([
                header,# ghi tiêu đề 
                '[{0' + space_fmt + '}/{1}]',# đây là chuỗi định dạng phức tạp hơn nơi {0}, {1}
                # sẽ được thay thế bằng số thứ tự của lần lặp hiện tại tương ứng với độ rộng space_ftm = int
                'eta: {eta}', # thời gian ước tính còn lại để hoàn thành tất cả các lần lặp
                '{meters}',
                'time: {time}', # thời gian trung bình cho mỗi lần lặp
                'data: {data}', # là thời gian trung bình để tải dữ liệu cho mỗi lần lặp
                'max mem: {memory:.0f}' #là lượng bộ nhớ tối đa sử dụng 
            ])
        else:
            # nếu như cuda không khả dụng bỏ qua việc sử dụng bộ nhớ tối đa
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        # Định nghĩa một metabyte để tính toán bộ nhớ 
        MB = 1024.0 * 1024.0 
        # Duyệt qua 1 danh sách iterable : 
        for obj in iterable: 
            # cập nhật thời gian tải dữ liệu dựa trên thời gian kết thúc vòng lặp trước
            data_time.update(time.time() - end)
            # sử dụng yield để trả về phần tử hiện tại của iterable 
            yield obj 
            # Cập nhật thời gian lặp hết 1 iteration 
            iter_time.update(time.time() - end)
            # kiểm tra xem biến đếm i (i ghi lại thứ tự số lần lặp là 1 giá trị int)
            # i có chia hết cho print_freq( là biến chỉ định tần suất in ra thông tin)
            # hoặc i = len(iteration) - 1
            if i% print_freq == 0 or i == len(iterable) -1 : 
                # tính toàn lại thời gian ước tính còn lại (eta) để hoàn thành 
                # tất cả các lần lặp dựa trên  thời gian hiện tại và thời gian kết thúc 
                # lần lặp trước
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # kiểm tra xem CUDA có khả dụng hay không 
                if torch.cuda.is_available():
                    # nếu cuda khả dụng in ra mà hình thông tin của 1 chuỗi định dạng
                    # log_msg theo định dạng đã được xác định trước đó với các tham  số được truyền vào 
                    print(log_msg.format(
                        # i chỉ số lần lặp , thời gian ước tính eta 
                        i, len(iterable), eta=eta_string,
                        # chuỗi các giá trị meters sủ dụng để làm mịn các giá trị trong 
                        # định dnagj này được nối bởi delimiter.
                        meters=str(self),
                        # thười gian kết thúc vòng lặp và thời gian tải dữ liệu
                        time=str(iter_time), data=str(data_time),
                        # Tính toán bộ nhớ được sử dụng 
                        memory=torch.cuda.max_memory_allocated() / MB))
                # trường hợp khi cuda không khả dụng
                else:
                    print(log_msg.format(
                        # i chỉ số lần lặp , thời gian ước tính eta 
                        i, len(iterable), eta=eta_string,
                        # chuỗi các giá trị meters sủ dụng để làm mịn các giá trị trong 
                        # định dnagj này được nối bởi delimiter.
                        meters=str(self),
                        # thười gian kết thúc vòng lặp và thời gian tải dữ liệu
                        time=str(iter_time), data=str(data_time),
                    ))

            # sau khi kết thúc 1 vòng lặp tăng chỉ số vòng lặp i lên 1 
            i += 1 
            # đồng thời cập nhật lại thời gian kết thúc của 1 vòng lặ iterable 
            end = time.time()
        # tính toán tổng thời gian cho 1 lần lặp 
        total_time = time.time() - start_time
        # biến đổi giá trị tổng thời gian này thành 1 dạng chuỗi để có thể in ra màn hình 
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

# Thiết lập phương thức get_sha 
# phương thức này trả về 1 chuỗi thông báo chứa mã SHA của commit hiện tại , 
# trạng thái của kho lưu trữ 
        
# Trong phương thức xử lý SHA , OS.PATH.DIRNAME được sử dụng để lấy tên của 
# một thư mục đường dẫn đã cho 
# và module subprocess chp phéo tạo và quản lý các tiến trình mới từ mã Python 
# của mình. Nó cung cấp các phương thức để kết nối với các ống dẫn nhập / xuất /lỗi 
# của tiến trình và lấy mã trạng thái trả về của tiến trình đó
def get_sha():
    # lấy ra tên của đường dẫn thư mục gán cho cwd 
    cwd = os.path.dirname(os.path.abspath(__file__))
    # Thiết lập phương thức def _run(cmd) để chạy 
    # một đường dẫn thư mục cwd (lệnh git)
    def _run(command):
        # lấy kết quả giải mã từ ascii sang mã python 
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    
    # Khởi tạo các biến với giá trị mặc định
    sha = 'N/A'  # Mã SHA của commit hiện tại
    diff = "clean"  # Trạng thái của các thay đổi (có thay đổi chưa commit hay không)
    branch = 'N/A'  # Nhánh hiện tại của kho lưu trữ

    try: 
        # lấy mã sha của commit hiện tại bằng cách chạy lệnh git 
        # với rev-parse : Một sy=ubcommand trong Git được sử dụng để chuyển đổi 
        # các tên (như nhánh hoặc tag) thành 1 mã băm SHA-1 tương ứng 
        # và Head:  1 Tham chiếu đến commit gần nhất bạn đang làm việc trên nhánh hiện
        # tại . Khi bạn sử dụng HEAD, Git sẽ trả về mã băm SHA-1 của commit đó.
        sha = _run(['git', 'rev-parse', 'HEAD'])
        # kiểm tra xem có thay đổi nào chưa commit hay không 
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        # lấy trạng thái cuả các thay đổi 
        diff = _run(['git', 'diff-index', 'HEAD'])
         # Nếu có thay đổi chưa commit, cập nhật biến diff
        diff = "has uncommited changes" if diff else "clean"
        # lấy nhánh hiện tại 
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        # Nếu có lỗi xảy ra, bỏ qua và giữ nguyên giá trị mặc định
        pass

    # Tạo thông báo với các thông tin đã lấy được
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

# Thiết lập phương thức collate_fn để chuân bị dữ liệu cho việc tải 
# lo batch trong quá trình huấn luyện cho mô hình . Nó chuyển đổi một danh 
# sách các mẫu dữ liệu thành một câú trúc dữ liệu phù hợp để xử lý hàng loạt 
# Trong trường hợp này nó chuyển đổi danh sách tensor đầu tiên thành một 
# nested tensor.. 
def collate_fn(batch):
    # chuyển đổi 1 danh sách các tuple thành một tuple của các dnah sách 
    # *batch dùng để unpack từng một tuple và chuyển chúng vào zip như là các đối 
    # số riêng biệt. Điều này cho phéo zip kết hợp từng phần tử tương ưng thành 1 tuple mới 
    batch = list(zip(*batch))
    # chuyển đổi danh sách tensor đầu tiên thành một nested_tensorm
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    # trả về 1 tuple chứa các dnah sách hoặc nested tensor 
    return tuple(batch)

# Thiết lập phương thức max_by_axis nhận vào 1 danh sách của các danh sách số nguyên 
# và trả về một dnah sách chứa giá trị lớn nhất cho mỗi trục axis 
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    # Khởi tạo danh sách maxes với giá trị của phần tử đầu tiên trong the_list
    maxes = the_list[0]
    # Duyệt qua từng danh sách con trong the_list từ phần tử thứ 2 đến hết danh sách 
    for sublist in the_list[1:]:
        # Duyệt qua từng phần tử trong danh sách con và cập nhật maxes
        # lấy ra chỉ số và giá trị của từng phần tử 
        for index, item in enumerate(sublist):
            # thêm vào danh sách maxes từng chỉ số vị trí 
            # sửt dụng hàm max để lấy ra các item có giá trị lớn nhất 
            # thêm náo vào dnah sách maxes theo chỉ số index 
            maxes[index] = max(maxes[index], item)
    # Trả về danh sách chứa giá trị lớn nhất theo từng trục
    return maxes

# Thiết lập lớp xử lý nested_tensor để tạo ra các tensor lồng nhau 
class NestedTensor(object):
    # thiết lập phương thức khởi tạo nhận đầu vào là một tensor 
    # và mask (một mặt nạ dưới dạng tùy chọn có thể là 1 tensor)
    def __init__(self, tensors, mask:Optional[Tensor]):
        # khởi tạo một nested tensor và một mặ nạ tùy chọn mask 
        self.tensors = tensors 
        self.mask = mask 
    # Thiết lập phương thức to để phân bổ nested tensor cho thiết bị được 
    # chỉ định 
    def to(self, device):
        # chuyển nested tensor sang thiết bị được chỉ định 
        cast_tensor = self.tensors.to(device)
        # gán lại mask = thuộc tính mask 
        mask = self.mask 
        # kiểmtra 1 điều kiện nếu như mask không = None 
        if mask is not None: 
            # kiểm tra một điều kiện như trên một lần nữa 
            # để đảm bảo rằng mask có tồn tại 
            assert mask is not None 
            # sau nó chuyển nó sang thiết bị được chỉ định 
            cast_mask = mask.to(device)
        # trường hợp còn lại tức mask = None 
        else:
            # gán cast_mask = N0ne
            cast_mask = None
        # trả về kết qủa của chính Nested tensor với cast_tensor và cast_mask 
        return NestedTensor(cast_tensor, cast_mask)
    
    # Thiết lập phương thức decompose để phân dã nested_tensor 
    # thành tensors và mask 
    def decompose(self):
        return self.tensors, self.mask

    # sau đó thiết lập phương thức để định nghĩa cách mà nested tensor được hiển trị 
    def __repr__(self):
        return str(self.tensors)
    

# Thiết lập phương thức tạo các nested tensor từ danh sách các tensor 
# nhận đầu vào là tensor_list được chỉ định là 1 danh sách các tensor
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # Công việc làm cho điều này trở nên tổng quát hơn 
    # kiểm tra 1 điều kiện xem tensor đầu tiên từ danh sách đầu vào tensor list
    # có phải có số chiều = 3
    if tensor_list[0].ndim == 3 :
        # kiểm tra xem có đang trong quá trình truy tìm của torch version hay không 
        if torchvision._is_tracing():
            # hàm này không xuất tốt sang ONNX nên gọi hàm thay thế 
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        
        # Tính kích thước lớn nhất từ dnah sách các tensor hỗ trợ các hình ảnh có kích 
        # thước khác nhau . 
        # Hàm max_size được sử dụng để tìm kiếm kích thước lớn nhất trong tất cả các tensor, điều 
        # này hữu ích khi làm việc với các hình ảnh có kích thước khác nhau , khi 
        #  muốn tạo một NestedTensor mà không cần phải thay đổi kích thước của chúng.
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # xác định kích thước lô dựa trên số lượng tensor và kích thước lớn nhất của tensor 
        # nếu tensor lớn nhất shape = 32 , 32 , 3 và danh sách này có 4 tensor thì 
        # batch shape = [4 , 32 , 32 , 3]
        batch_shape = [len(tensor_list)] + max_size
        # giải nén kích thước lô thành các kích thước riêng lẻ .
        # các kích thước lần lượt đại diện cho số lượng tensor , số kênh và 2 chiều của tensor lô
        b, c, h, w = batch_shape 
        # lấy da kiểu dữ liệu của tensor đầu tiên trong dnah sách tensor_list 
        dtype = tensor_list[0].dtype 
        # và lấy ra thông tin thiết bị mà tensor đầu tiên được chỉ định 
        device = tensor_list[0].device  
        # tạo một tensor chứa các gía trị = 0 shape = kích thưovs của tensor batch_shape
        tensor = torch.zeros(batch_shape , dtype=dtype, device=device)
        # tạo một tensor mask shape = b , h , w bỏ qua kích thước channels vì đó chỉ là kích thước ảnh màu 
        # dtype = boolean nhằm mục đích che đi các giá trị không cần thiết giảm thiểu đi số 
        # lượng tham số không cần thiết 
        mask = torch.ones((b,h,w), dtype=torch.bool, device=device)
        # duyệt qua 1 danh sách tensor_list tensor biểu diễn , tensor có kích thước = [n , 3 * h*w]
        # sử dụng để đệm hàng loạt các tensor và mask là các mặt nạ tương ứng 
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # thực hiện sao chép từng tensor theo các chiều vào tensor lớn với padding
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # cập nhật mặt nạ để chỉ định phần nào của tenor là hợp lệ 
            m[: img.shape[1], :img.shape[2]] = False
    # trường hợp còn lại 
    else:
        # đưa ra một cảnh báo không hỗ trợ 
        raise ValueError('not supported')
    # cuối cùng trả về kết quả của nestedtensor với 2 giá trị tensor và mask
    return NestedTensor(tensor, mask)

# Xây dựng phương thức _onnx_nested_tensor_from_tensor_list là một 
# phiên bản của hàm nested_tensor_from_tensor_list được thiết kế để tương thích 
# với ONNX tracing (truy tìm ONNX)
# Trong PyTorch, ONNX (Open Neural Network Exchange) là một định dạng mở cho các mô hình học sâu,
# cho phép chúng được sử dụng trên nhiều nền tảng và công cụ khác nhau.
# hàm này nhận đầu vào là 1 danh sách tensor trả về 1 dạng nested tensor 
@torch.jit.unused  # Đánh dấu hàm này không được sử dụng trong quá trình JIT compilation
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    # khởi tạo 1 dnah sách để lưu trữ kích thước lớn nhất 
    max_size = []
    # duyệt qua từng chiều của tensor 
    for i in range(tensor_list[0].dim()):
        # duyệt qua dnah scahs các tensor trong tensor list theo chỉ số [i]
        # sau đó nối các giá trị này dạng stack chuyển nó dạng float và sau đó là int 
        # sử dụng max lấy ra giá trị lớn nhất và gán chp max_size_ i
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        # thêm giá trị lớn nhất được lấy theo từng chiều vào danh sách max_size 
        max_size.append(max_size_i)
    # Chuyển max_size thành 1 tuple để các giá trị của nó là mặc định không thể thay đổi 
    max_size = tuple(max_size)

    # tiếp theo thực hiện việc tạo nested tensor 
    # tạo 2 danh sách 
    # padded_imgs để lưu các tensor đã được đệm 
    padded_imgs = []
    # và padded_masks để lưu trữ các mask đã được đệm 
    padded_masks = []
    # duyệt qua dnah sách các tensor biểu diễn 
    for img in tensor_list: 
        # tính toán từng phần kích thước thiếu của các tensor biểu diễn vơi 
        # các giá trị trong tuple max_size tương ứng . Sử dụng zip để phân giải 
        # các giá trị 
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        # sau đó thêm đệm theo các chiều của tensor biểu diễn các giá trị đệm gán  = 0
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        # thêm hình ảnh (tensor biểu diễn) đã được đệm vào dnah sách chứa tương ứng 
        padded_imgs.append(padded_img)

        # tạo một tensor m kích thước = img[0] và được sử lý bởi thiết bị chỉ định 
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        # Truyền vào kích thước m cho mask để có kích thước = tensor biểu diễn phục vụ cho việc masking
        # và đặt các giá trị trong mask không thể thay đổi
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        # thêm mask đã được đệm vào dnah sách và đặt các giá trị trong mask = Boolean 
        padded_masks.append(padded_mask.to(torch.bool))
    # cuối cùng nối các tensor mask và tensor_padd lại với nhau dưới dạng stack 
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    # cuối cùng trả về đối tượng nested tensor duy nhất của tensor và mask 
    return NestedTensor(tensor, mask=mask)


# Thiết lập phương thức setup_for_distributed 
# có chức năng vô hiệu hóa việc in ra khi không ở trong quy trình chính 
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__ # Nhập mô-đun builtins để tham chiếu đến hàm print gốc
    builtin_print = __builtin__.print # Lưu trữ tham chiếu đến hàm print gốc

    # thiết lập phuuwpng thức print  # Định nghĩa hàm print mới
    def print(*args, **kwargs):
        # Xóa và trả về giá trị của 'force' từ kwargs, mặc định là False
        force = kwargs.pop('force', False)
        # nếu như đang ở quy trinh chính thì cho phép in ra 
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print  # Ghi đè hàm print gốc bằng hàm print mới
    

# Xây dựng môi trường phân tán (hay là chế độ phân tán cho môi trường huấn luyện)
def init_distributed_model(args):
    # Kiểm tra xem các biến môi trường là Rank thứ tự của quá trình tham gia 
    # vào việc huấn luyện phân tán . 
    # World_size là tổng số quá trình tham gia vào cuộc huấn luyện (số thiết bị)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # nếu có thiết lập rank và world cho quá trình hiện tại từ các biến môi trường
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_sIZE'])
        # Và thứ tự của GPU trong quá trình hiện tại (hay được gán cho quá trình hiện tại)
        args.gpu = int(os.environ['LOCAL_RANK'])
    # KIỂM tra xem slurm_procid có tồn tại hay không 
    # SLURM_PROCID là một biến môi trường được sử dụng bởi SLURM một hệ thống quản lý 
    # hàng đợi công việc, để xác định ID của quá trình. 
    elif 'SLURM_PROCID' in os.environ: 
        # Nếu có lấy ra ID trong hàng đợi quá trình 
        args.rank = int(os.environ['SLURM_PROCID'])
        # và GPU của quá trình 
        args.gpu = args.rank % torch.cuda.device_count()
    # nếu như không có các biế môi trường cần thiết
    else:# Thông báo không sử dụng môi trường phân tán 
        print('Not using distributed mode')
        # Đặt môi trường phân tán = False
        args.distributed = False
        return

    # gán lại giá trị cho môi trường phân tán = True (có tồn tại)
    args.distributed = True
    # Thiết lập GPU với tham số args.gpu cho quá trình, tên thiết bị cụ thể được sử 
    # dụng cho quá trình hiện tại dựa trên Local_rank cho thứ tự của các GPU 
    torch.cuda.set_device(args.gpu)
    # Thiết lập backend phân tán là 'nccl' là một backend được hỗ trợ cho việc 
    # giao tiếp hiệu quả với GPU . 
    args.dist_backend = 'nccl'
    # in ra mà hình console một thông tin theo định dạng gồm 
    # rank thứ tự của quá trình hiện tại (hoặc thứ tự trong hàng đợi)
    # và dist_url là 1 chuỗi kết nối có thể chỉ định đến 1 địa chỉ máy 
    # hoặc chỉ định URL cho việc khởi tạo nhóm quá trình phân tán 
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    
    # khởi tạo nhóm cho qúa trình phân tán  với thông tin backend, phương thức khởi tạo
    # kích thước world_size (ở đây world_Size đại diện cho số lượng tiến trình)
    # rank (thứ tự các tiến trình tham gia)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # Sau đó tạo một dào cản ngặn chặn đến khi tất cả các tiến trình hay nhóm tiến trình đều đã 
    # được khởi tạo xong.
    torch.distributed.barrier()
    # cuối cùng gọi hàm setup_for_distributed với tham số chỉ ra liệu tiến chình hiện tại 
    # có phải là quy trình chính hay không. nêu nó là chính rank = 0. và cho phép in 
    # thông điệp ra màn hình.
    setup_for_distributed(args.rank == 0)

# Thiết lập phương thức accuracy để tính toán đo lường độ chính xác tại top_k
# và với @torch.np_grad để bỏ qua việc tính toán gradient descent
@torch.no_grad()
def accuracy(output, target,topk=(1,)):
    # Top k là một tuple chứa các giá trị k mà muốn tính toán độ chính xác 
    # output là tensor chứa các dự đoán của mô hình 
    # target tensor chứa các nhãn đúng
    """computes the precision@k for the specified values of k.
        Tính toán độ chính xác cho các giá trị cụ thể của k.
    """
    # kiểm tra xem số lượng các phần tử trong danh sách target = none tức target= rỗng 
    if target.numel() == 0:
        # trả về giá trị 0 được xử lý bởi gpu 
        return [torch.zeros([], device=output.device)]
    # lấy k có giá trị lớn nhất trong tuple topk để thực hiện so sánh độ chính xác của 
    # giá trị này với thực tế. 
    maxk = max(topk)
    # lấy ra kích thước lô từ tensor đầu vào 
    batch_size = target.size(0)
    # lấy ra top_k dự đoán từ output 
    # tham số thứu nhất là nguồn , tham số thứ hai = 1 tức là chiều muốn trích xuất gia trị '
    # chiều cột , và tham số thứ ba = True là cho phép các gia trị sắp xếp giảm dần . 
    # Tham số cuối cùng = True để trả về giá trị và chỉ số của maxk giá trị hàng đầu .
    # kết quả là 2 tensor 1 chứa chỉ số và 1 tensor chứa giá trị 
    _, pred = output.topk(maxk, 1 , True , True)
    pred = pred.t()  # Chuyển vị pred để so sánh với target
    # So sánh dự đoán với target và tạo tensor correct chứa kết quả
    # sử dụng pred.eq để so sánh tensor target và pred 
    # target.view(1, -1) định hình lại tensor target shape = [1* n]
    # sử dụng expand_as(pred) để mở rộng tensor target để nó có cùng kích thước với tensor pred 
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []  # Danh sách để lưu kết quả
    for k in topk:
        # Tính số lượng dự đoán chính xác tại top-k
        # trích xất từ 0-> đến k hàng đầu tiên sau đó làm phẳng thành tensor 1 chiều
        correct_k = correct[:k].view(-1).float().sum(0)
        # Tính tỷ lệ phần trăm và thêm vào danh sách kết quả
        res.append(correct_k.mul_(100.0 / batch_size))
    # Trả về danh sách tỷ lệ phần trăm độ chính xác tại top-k
    return res  

# Xây dựng một hàm nội suy interpolate thực hiện chức năng nội suy mẫu 
# là quá trình thay đổi kích thước mẫu up hoặc down tùy thuộc vào tham số truyền vào 
# type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # Kiểm tra phiên bản của torchvision
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        # Nếu tensor đầu vào không rỗng
        if input.numel() > 0:
            # Sử dụng hàm interpolate của PyTorch với các tham số (scale_factor là hệ số nội suy nếu > 1 up and dow < 1)
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )
        # Nếu tensor đầu vào rỗng 
        # Hàm _output_size được gọi với bốn tham số: số chiều không gian (trong trường hợp này là 2 cho tensor 2D),
        # tensor đầu vào input, kích thước đầu ra mong muốn size, và tỷ lệ nội suy scale_factor.
        output_shape = _output_size(2, input, size, scale_factor)
        # lấy tất cả ác chiều từ tensor input trừ 2 chiều cuối 
        output_shape = list(input.shape[:-2]) + list(output_shape)
        # Tạo một tensor mới rỗng với kích thước đã được tính toán
        return _new_empty_tensor(input, output_shape)
    else:
        # Nếu phiên bản torchvision mới hơn, sử dụng hàm interpolate từ torchvision
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)