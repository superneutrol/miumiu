import click 
import torch 
import torchvision.transforms as T 
from functools import reduce 
from pathlib import Path 

from dalle2_pytorch import DALLE2, Decoder, DiffusionPrior 

# Thiết lập phương thức safeget sử dụng để trả về một giá trị tương ứng với khóa hoặc 
# giá trị mặc định nếu không tìm thấy khóa trong từ điển 
def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)


# Thiết lập phương thức simple_slugify nhận vào một chuỗi văn bản và độ dài tối đa
# sau đó thay thế các ký tự đặc biệt bằng dấu gạch dưới và cắt chuỗi theo độ dài tối đa 
def simple_slugify(text, max_length = 255):
    return text.replace("-", "_").replace(",", "").replace(" ", "_").replace("|", "--").strip('-_')[:max_length]


# get_pkg_version trả về phiên bản của thư viện dalle2_pytorch đang được sử dụng 
def get_pkg_version():
    from pkg_resources import get_distribution
    return get_distribution('dalle2_pytorch').version

def main():
    pass


# sử dụng click để tạo một dòng lệnh
# 1: click.command() đây là một trang trí (decorator) của thư viện click dùng để định nghĩa một lệnh mới
@click.commnad()
# 2: click.option('--model--'....) định nghĩa một tùy chọn model cho lệnh với giá trị mặc định 
# là './dalle2.pt' Tùy chọn này dùng để chỉ đường dẫn đến mô hình DALL-E2 đã được huấn luyện.
@click.option('--model', default = './dalle2.pt', help = 'path to trained DALL-E2 model')
# 3: click.option('--cond_scale'  Định nghĩa một tùy chọn --cond_scale cho lệnh, với giá trị mặc định là 2. 
# Tùy chọn này dùng để chỉ tỷ lệ điều kiện (conditioning scale) trong bộ giải mã (decoder).
@click.option('--cond_scale', default = 2, help = 'conditioning scale (classifier free guidance) in decoder')
# 4: @click.argument('text'): Định nghĩa một đối số text cho lệnh. Đối số này sẽ được truyền vào hàm dream.
@click.argument('text')
def dream(
    model,
    cond_scale,
    text
):
    # Tạo một đối tượng Path từ thư vuện pathlib với đường dẫn là giá trị của tham số 
    # model 
    model_path = Path(model)
    # Lấy đừơng dẫn tuyểh đối của model_path và chuyển thành chuỗi 
    full_model_path = str(model_path.resolve()
                          )
    
    # kiểm tra xem đường dẫn model_path có tồn tại hay không
    # nếu không nó sẽ nem ra một thông báo 
    assert model_path.exists(), f'model not found at {full_model_path}'
    # sử dụng hàm load từ thư viện torch để tải mô hình từ đường dẫn model_path
    loaded = torch.load(str(model_path))

    # sử dụng hàm safeget được định nghĩa ở trên để lấy giá trị của khóa 'version' từ từ điển loader
    version = safeget(loaded, 'version')
    # In ra thông tin về việc tải mô hình 
    print(f'loading DALL-E2 from {full_model_path}, saved at version {version} - current package version is {get_pkg_version()}')

    # Sử dụng hàm safeget để lấy giá trị của khóa ‘init_params.prior’ từ từ điển loaded. 
    # Giá trị này được lưu vào biến prior_init_params.
    prior_init_params = safeget(loaded, 'init_params.prior')
    #  Tương tự như trên, lấy giá trị của khóa ‘init_params.decoder’ từ từ điển loaded và lưu vào biến decoder_init_params.
    decoder_init_params = safeget(loaded, 'init_params.decoder')
    # Lấy giá trị của khóa ‘model_params’ từ từ điển loaded và lưu vào biến model_params.
    model_params = safeget(loaded, 'model_params')


    #  Khởi tạo một đối tượng DiffusionPrior với các tham số được truyền vào từ prior_init_params.
    prior = DiffusionPrior(**prior_init_params)
    # Khởi tạo một đối tượng Decoder với các tham số được truyền vào từ decoder_init_params
    decoder = Decoder(**decoder_init_params)

    # : Khởi tạo một đối tượng DALLE2 với prior và decoder đã được khởi tạo ở trên.
    dalle2 = DALLE2(prior, decoder)
    # Tải trạng thái của mô hình từ model_params vào đối tượng dalle2.
    dalle2.load_state_dict(model_params)

    # Gọi phương thức của đối tượng dalle2 với đối số là text và cond_scale. Kết quả trả về được lưu vào biến image.
    image = dalle2(text, cond_scale = cond_scale)

    # : Chuyển đổi image thành định dạng PIL (Python Imaging Library) và lưu vào biến pil_image.
    pil_image = T.ToPILImage()(image)
    #  Lưu pil_image thành một file PNG với tên file được tạo từ text thông qua hàm simple_slugify.
    return pil_image.save(f'./{simple_slugify(text)}.png')