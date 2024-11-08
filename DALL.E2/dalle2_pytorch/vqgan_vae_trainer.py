import math 
from math import sqrt 
import copy 
from random import choice 
from pathlib import Path 
from shutil import rmtree 
from PIL import Image 

import torch 
from torch import nn 
from torch.cuda.amp import autocast, GradScaler 
from torch.utils.data import Dataset, DataLoader, random_split 

import torchvision.transforms as T 
from torchvision.datasets import ImageFolder 
from torchvision.utils import make_grid, save_image

from einops import rearrange 

from dalle2_pytorch.vqgan_vae import VQGanVAE 
from dalle2_pytorch.optimizer import get_optimizer

from ema_pytorch import EMA 

# Xây dựng các hàm chức năng 
def exists(val):
    # hàm này trả về chính tensor này nếu như nó có tồn tại 
    return val is not None 

# Xây dựng hàm snoop là một hàm no-operator nó không thực hiện bất kỳ 
# hành động nào và luôn trả về None 
def noop(*args, **kwargs):
    pass

# Xây dựng hàm cycle hàm này thể hiện một Trình tự 
# nó là một generator vô hạn 
def cycle(dl):
    while True: 
        for data in dl : 
            # Trả về từng phần tử một cho đến cuối dnah sách 
            # và nó sẽ bắt đầu lại từ đầu 
            yield data 


# Xây dựng hàm cast tuple hàm này sẽ biến đổi t thành một tuple 
def cast_tuple(t): 
    # hàm này sẽ trả về tuple hoặc list nếu t là một trong 2 kết quả này 
    # nếu không nó sẽ tạo ra một tuple mới chứa t 
    return t if isinstance(t, (tuple, list)) else (t,)


# Xây dựng hàm yes or no hiển thị một câu hỏi và yêu cầu người dùng 
# nhập câu trả lời 
def yes_or_no(question):
    # yêu cầu một câu trả lời y/n 
    answer = input(f'{question} (y/n)')
    # trả về True nếu anser yes, y và sẽ không phân biệt chữ hoa hau thường 
    # còn không return False 
    return answer.lower() in ('yes', 'y')


# Xây dựng hàm accum_log hàm này sẽ cập nhật giá trị trong từ điển 
# log bằng cách cộng dồn các giá trị trong từ điển new_logs 
def accum_log(log, new_logs):
    # duyệt qua danh sách các cặp keys values trong từ điển new_logs 
    for key, new_value in new_logs.items():
        # lấy ra các values trong từ điển log với key tương ứng nếu values của key
        # không tồn tại gán nó  = 0
        old_value = log.get(key, 0)
        # sau đó cập nhật value cho key hiện tại của từ điển log bằng cách cộng dồn 
        # old_value và new_vale 
        log[key] = old_value + new_value

    return log # trả về từ điển log 


# Xây dựng lớp ImageDataset lớp này có chức năng 
# đọc các hình ảnh từ tập folder và áp dụng phép tăng cường hình ảnh 
class ImageDataset(Dataset):
    # Thiết lập phương thức khởi tạo 
    def __init__(self, folder, image_size, exts = ['jpg','jpeg', 'png']):
        super().__init__()
        # đinh nghĩa một thuộc tính folder là tệp thư mục chứa các đường dẫn hình ảnh 
        self.folder = folder 
        # định nghĩa thuộc tính là kích thước của hình ảnh 
        self.image_size = image_size
        # Định nghĩa một danh sách path danh sách này sẽ chứa 
        # các đường dẫn thỏa mãn định dạng của danh sách exts 
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # In ra độ dài của danh sách path 
        print(f'{len(self.paths)} training samples found at {folder}')

        # xÂY DỰNG một danh sách kết hợp các lớp biến đổi hình ảnh
        self.transform = T.Compose([
            # ĐÂU TIÊN định dạng nó thành ảnh RGB
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            # SAU ĐÓ RESIZE VỀ KÍCH THƯỚC TIÊU CHUẨN
            T.Resize(image_size),
            # lật ngang hình ảnh 
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    # Xây dựng hàm len hàm này trả về độ dài danh sách paths
    def __len__(self):
        return len(self.paths)

    # Hàm trả về các hình ảnh và hình ảnh được tăng cường
    def __getitem__(self, index):
        # lấy ra một đường dẫn từ path theo index
        path = self.paths[index]
        # sau đó sử dụng PIL.Image để đọc hình ảnh từ đường dẫn 
        img = Image.open(path)
        # áp dụng phép biến đổi transform lên các hình ảnh
        return self.transform(img)
    


# xÂY DỰNG LỚP ĐÀO TẠO CHÍNH CHO MÔ HÌNH VECTOR QUANTIZATION Gan
class VAGanVAETrainer(nn.Module):
    # Thiết lập phương thức khởi tạo 
    def __init__(self, vae, *, num_train_steps, lr,
        batch_size,  folder, grad_accum_every, wd = 0.,
        save_results_every = 100, save_model_every = 1000,
        results_folder = './results', valid_frac = 0.05,
        random_split_seed = 42,  ema_beta = 0.995,
        ema_update_after_step = 500, ema_update_every = 10,
        apply_grad_penalty_every = 4, amp = False  ):

        super().__init__()
        # đảm bảo rằng thuộc tính vae phải là một instance của VQGenVAE 
        assert isinstance(vae, VQGanVAE), 'vae must be instance of VQGanVAE'
        # lấy ra kích thước image_size từ model VAE 
        image_size = vae.image_size

        # định nghĩa thuộc tính vae 
        self.vae  = vae 
        # khở tạo một đối tượng EMA đê theo giõi phiên bản trung bình di chuyển 
        # của vae 
        self.ema_vae = EMA(model= vae, update_after_step=ema_update_after_step, update_every=ema_update_every)

        # đưng ký một bộ đệm không được cập nhật trong qúa trình học tập steps với 
        # giá trị khởi tạo = 0
        self.register_buffer("steps", torch.Tensor([0]))

        # ĐỊNH nghĩa số lượng bước huấn luyện của mô hình
        self.num_train_steps = num_train_steps
        # kích thước lô batch_size 
        self.batch_size = batch_size
        # và Tần suất tích lũy gradient descent 
        self.grad_accum_every = grad_accum_every

        # lấy tất cả tham số của vae và lưu chúng vào set 
        all_parameters = set(vae.parameters())
        # tương tự lấy toàn bộ tham số của discriminator 
        discr_parameters = set(vae.discr.parameters())
        # Tính toán các tham số của VAE bỏ đi  phần tham số của bộ phân loại 
        vae_parameters = all_parameters - discr_parameters

        # Định nghĩa trình tối ưu hóa với tham số vae , leaning_rate và 
        # weight_decay 
        self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
        # Và định nghĩa một trình tối ưu hóa cho Discriminator với các tham số tương ứng 
        self.discr_optim = get_optimizer(discr_parameters, lr = lr, wd = wd)


        # Thiết lập sử dụng kỹ thuật MIXED Precision Training (đào tạo độ chính xác hỗn hợp )
        # Đinnhj nghĩa thuộc tính AMp 
        self.amp = amp 
        # Khởi tạo các GradScaler để điều chỉnh tỷ lệ của gradient giúp tôi ưu hóa qúa 
        # trình huấn luyện với độ chính xác hỗn hợp 

        # Create dataset 
        # định nghĩa một dataset 
        self.ds = ImageDataset(folder, image_size=image_size)

        # Thực hiện việc phân chia các phần dữ liệu cho các nhiệm vụ khác nhau 

        # nếu như hệ số valid_fact > 0 hệ số này đại diện cho một tỷ lệ 
        #  > 0 và < 1 được sử dụng để lấy một số lượng theo % dữ liệu 
        if valid_frac > 0:
            # Tính toán kích thước cho dữ liệu huấn luyện 
            train_size = int((1 - valid_frac) * len(self.ds))
            # khi có được kích thước cho dữ liệu đào tạo ta tính kích
            # thước dữ liệu validation 
            valid_size = len(self.ds) - train_size
            # sử dụng hàm random_split để tách danh sách self.ds thành 2 phần valid và train với kích thước được chỉ định 
            # torch.Generator().manual_seed(random_split_seed đảm bảo rằng việc chia dữ liệu là 
            # nhất quán giữa các lần chạy khác nhau nếu cùng một random_split_seed được sử dụng 
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            # In ra thông tin về độ dài của 2 danh sách dữ liệu này 
            print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        
        # trường hợp còn lại nếu hệ số này không tồn tại 
        else:
            # ta gán valid_ds bằng toàn bộ dnah sách 
            self.valid_ds = self.ds
            # sau đó in ra thông tin về danh sách dữ liệu 
            print(f'training with shared training and valid dataset of {len(self.ds)} samples')


        # Định nghĩa 2 thuộc tính dl và valid_dl được 
        # sử dụng để tải dữ liệu từ danh sách self.valid_ds và ds_train 

        self.dl = cycle(DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        ))
        # Hàm cycle được sử dụng để lấy mẫu liên tục giữ liệu 
        # để đảm bảo trong các kỷ nguyên huân luyện không cần gọi nhiều lần thuộc tính này 
        # hàm cycle là một hàm trình tự khu lặp hết một danh sách nó sẽ lặp lại danh sách đó từ đầu 
        self.valid_dl = cycle(DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        ))

        # Khởi tạo thuộc tính save_model thuộc tính này xác định sau bao nhiêu bước 
        # huấn luyện thì mô hình sẽ được lưu lại 
        self.save_model_every = save_model_every
        # và thuộc tính save_result sẽ lưu lại kết quả đó 
        self.save_results_every = save_results_every

        # Thiết lập biến để xác định sau bao nhiêu bước huấn luyện thì hình phạt gradient sẽ được áp dụng.
        self.apply_grad_penalty_every = apply_grad_penalty_every

        # Khởi tạo đối tượng Path với đường dẫn đến thư mục kết quả, 
        # cho phép thao tác dễ dàng với hệ thống tệp.
        self.results_folder = Path(results_folder)

        # iểm tra xem thư mục kết quả có chứa tệp hay thư mục con không. 
        # Nếu có, hỏi người dùng liệu họ có muốn xóa các điểm kiểm tra và kết quả của thí nghiệm trước đó không.
        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            # Nếu người dùng chọn “có”, hàm rmtree sẽ được sử dụng để xóa toàn bộ thư mục kết quả.
            rmtree(str(self.results_folder))

        # Tạo thư mục kết quả nếu nó chưa tồn tại. parents = True có nghĩa là nếu thư mục cha không tồn tại, 
        # nó cũng sẽ được tạo. exist_ok = True có nghĩa là nếu thư mục đã tồn tại, không có ngoại lệ nào sẽ được ném ra.
        self.results_folder.mkdir(parents = True, exist_ok = True)


    # Xây dựng phương thức train_step thực hiện xử lý tại mỗi bước đào tạo 
    # mô hình 
    def train_step(self):
        # lấy ra tên thiết bị từ thiết bị được sử dụng bởi mô hình vae 
        device = next(self.vae.parameters()).data
        # lấy ra các buóc thời gian 
        steps = int(self.steps.item())
        # thiết lập hình phạt gradient nếu như step % số lượng các gradien_penalty được áp dụng
        # thì thm số này được thiết lập 
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        # Thiết lập chế độ train cho mô hình val 
        self.vae.train()

        # Một từ điển log lưu trữ các nhật ký 
        logs = {}

        # Update vae (generator)

        # Một vòng lặp được thực hiện self.grad_accum_every lần đại diện cho số lần 
        # tích lũy gradient trước khi thực hiện một bước cập nhật 
        for _ in range(self.grad_accum_every):
            # Lấy hình ảnh tiếp theo từ bộ dữ liệu 
            img = next(self.dl)
            # chuyển hình ảnh đến thiết bị để thực hiện tính toán 
            img = img.to(device)

            # Xây dựng khối with autocast(enable = self.amp) cho phép sửu dụng 
            # độ chính xác hỗn hợp nếu self.amp được kích hoạt 
            with autocast(enable = self.amp):
                # Tính toán hàm mất mát của vae dựa trên độ chính xác hỗn hợp 
                loss = self.vae(
                    # yêu cầu vae trả về loss và áp dụng hình phạt gradient cho quá trình lan truyền 
                    img, return_loss = True, apply_grad_penalty= apply_grad_penalty
                )            

                # áp dụng quy tắc chuôi để tính toán gradient của los, được chi cho 
                # số lượng (số lần) gradient được tích lũy 
                self.scaler.scale(loss / self.grad_accum_every).backward()

            # Ghi lại giá trị loss trung bình vào nhật ký log 
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        
        # sử dụng self.scale.step để thực hiện một bước cập nhật trên bộ tối ưu hóa 
        # với các gradient đã được điều chỉnh 
        self.scaler.step(self.optim)
        # Cập nhật GradScaler để điều chỉnh tỷ gradien giúp tối ưu hóa quá trình huấn 
        # luyện với độ chính xác hỗn hợp 
        self.scaler.update()
        # Xóa các gradient hiện tại để chuẩn bị cho lần tính toán gradient tiếp theo 
        self.optim.zero_grad()


        # Cập nhật Discriminator 
        if exists(self.vae.discr):
            # gán discriminator loss =  0
            discr_loss = 0
            # duyệt qua một vòng lặp với số lần tích lũy gradeint 
            for _ in range(self.grad_accum_every):
                # Lâyd ra hình ảnh tiếp theo
                img = next(self.dl)
                # chuyển hình ảnh đó cho thiết bị GPU 
                img = img.to(device)

                # xÂY DỰNG KHỐI with autocast cho phép sử dụng độ chính xác hỗn 
                # hợp nếu như self.amp được kích hoạt 
                with autocast(enabled = self.amp):
                    # Yêu câu mô hình trả về los của Discriminator
                    loss = self.vae(img, return_discr_loss = True)

                    # áp dụng quy tắc chuỗi để tính toán lan chuyền ngược 
                    # dưạ trênn loss với số lần gradient được tích lũy 
                    self.discr_scaler.scale(loss / self.grad_accum_every).backward()

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            # sử dụng self.discr_scaler.step để thực hiện một bước cập nhật gradient dựa trên 
            # bộ tối ưu hóa với các gradient đã được điều chỉnh 
            self.discr_scaler.step(self.discr_optim)
            # Cập nhật GradScaler để điều chỉnh tỷ lệ của Gradient giúp tối ưu hóa quá 
            # trình huấn luyện với độ chính xác hỗn hợp 
            self.discr_scaler.update()
            # Xóa bỏ đi gradient để chuẩn bị cho lần tính toán gradient tiếp theo 
            self.discr_optim.zero_grad()

            # log
            # In ra thông tin về những gì được ghi trong nhật ký tại bước thời gian hiện tại
            print(f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")


        # Cập nhật mô hình Vae với trung bình di chuyển mũ EMA và lưu mẫu kết 
        # qủa sau một số bước nhất định 
        # update exponential moving averaged generator

        self.ema_vae.update()

        # sample results every so often
        # kiểm tra xem số bước hiện tại có chia hết cho 
        # số lượng kết quả sẽ được lưu trữ sau n bước hay không 
        if not (steps % self.save_results_every):
            # nếu chia hết tức là đã đến thời điểm lưu mẫu kết quả 
            
            # lặp qua 2 mô hình EMA và VAE gốc
            for model, filename in ((self.ema_vae.ema_model, f'{steps}.ema'), (self.vae, str(steps))):
                # chuyển đổi mô hình sang chế độ đánh giá 
                model.eval()

                # Lây một hình ảnh tiếp theo từ dữ liệu
                imgs = next(self.dl)
                # và chuyển nó cho thiết bị để tính toán
                imgs = imgs.to(device)

                # Tái tạo lại hình ảnh từ ảnh đầu vào 
                # gán kết quả cho recons
                recons = model(imgs)
                # nrows được tính toán để xác định số hàng trong lưới hình ảnh, dựa trên căn bậc hai của kích thước lô.
                nrows = int(sqrt(self.batch_size))

                # Thực hiện xếp chồng các ảnh gốc và hình ảnh tái tạo thành một tensor mới 
                imgs_and_recons = torch.stack((imgs, recons), dim = 0)
                # Xắp xếp lại tensor để hình ảnh gốc và ảnh tái tạo sen kẽ nhau 
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                # Chuyển tensor về CPU, loại bỏ thông tin gradient, chuyển sang dạng số thực 
                # và giới hạn giá trị pixel với .clam(0., 1.) trong khoảng từ 0 đến 1.
                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                # Tạo một lưới hình ảnh từ tensor để hiển thị.
                grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))
                
                # Lưu lưới hình ảnh vào nhật ký dưới khóa ‘reconstructions’.
                logs['reconstructions'] = grid

                # Lưu lưới hình ảnh vào thư mục kết quả với tên tệp được chỉ định.
                save_image(grid, str(self.results_folder / f'{filename}.png'))

            # In thông báo với số bước và đường dẫn thư mục nơi hình ảnh được lưu.
            print(f'{steps}: saving to {str(self.results_folder)}')
        

         # save model every so often
        # kiểm tra xem số lượng bước hiện tại có chia hết cho 
        # số lượng bước được chỉ định để lưu mô hình hay không 
        if not (steps % self.save_model_every):
            # Lấy trạng thái hiện tại của mô hình VAE
            # bao gồm tất cả các tham số và trọng số 
            state_dict = self.vae.state_dict()
            # Model path được tạo ra để xác định vị trí lưu mô hình 
            # với tên tệp bào gồm số bước hiện tại 
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            # lưu trữ các tham số , trọng số của mô hình vào đường dẫn model_path 
            torch.save(state_dict, model_path)

            # Qúa trình được lặp lại với mô hình ema_VAE 
            ema_state_dict = self.ema_vae.state_dict()
            # tạo đường dẫn model Path để xác định vị trí lưu mô hình
            model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
            # Thực hiện việc lưu trữ mô hình 
            torch.save(ema_state_dict, model_path)

            # in ra thông báo rằng mô hình đã được lưu vào thư mục kết quả 
            print(f'{steps}: saving model to {str(self.results_folder)}')
        
        # cập nhật bươc thời gian lên 1
        self.steps += 1
        # và trả về từ điển logs 
        return logs
    

    # Xây dựng hàm train hàm này sẽ thực hiện các bước đào tạo mô hình 
    # nhận đầu vào tham số log_fn = noop là một phương thức 
    # phương thức này biểu thị cho việc trả ra tất cả tham số được định nghĩa
    def train(self, log_fn = noop): 
        # lấy ra thiết bị sử dụng GPU 
        device = next(self.vae.parameters()).device

        # SỬ DỤNG VÒNG LẶP WHILE ĐỂ XÁC ĐỊNH MỘT ĐIỀU KIỆN DÀI HẠN
        while self.steps < self.num_train_steps:
            # GÁN CHO TỪ ĐIỂN LOG BẰNG kết quả của hàm train_step
            logs = self.train_step()
            # sau đó gọi hàm log_fn hàm này sẽ trả về từ điển logs
            log_fn(logs)

        # sau khi vòng lặp while kết thúc in ra thông báo quá trình huấn luyện hoàn thành 
        print('training complete')




        




