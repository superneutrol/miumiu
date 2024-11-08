# Đây chỉ là xây dựng các hàm chức năng 
# thực hiện tải các tài nguyên hình ảnh và áp dụng việc tăng cường hình ảnh cho mô hình 
import pathlib 
from pathlib import Path 
import torch 
from torch.utils import data 
from torchvision import transforms, utils 

from PIL import Image 

# Xây dựng một phương thức chu trình 
# sẽ liên tục thực hiện một trình tự trả về các mẫu dữ liệu trong
# danh sách chứa nó 
# helpers functions

def cycle(dl):
    while True:
        for data in dl:
            yield data

# dataset and dataloader
class Dataset(data.Dataset):
    # thiết lập phương thức khởi tạo 
    def __init__(
            self, folder, image_size, exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        # định nghĩa các thuộc tính 
        self.folder = folder 
        self.image_size = image_size
        # lưu trữ tất cả các hình ảnh trong thư mục có thảo mãn 
        # 1 trong các định dạng file của danh sách exts
        self.path = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # Định nghĩa một danh sách các biến đổi cho hình ảnh 
        self.transform = transforms.Compose(
            [
                # 1: Resize hình ảnh về kích thước tiêu chuẩn 
                transforms.Resize(image_size),
                # 2: áp dụng lật hình ảnh 
                transforms.RandomHorizontalFlip(),
                # 3: cắt hình ảnh 
                transforms.CenterCrop(image_size),
                # 4: Biều diễn các hình ảnh dưới dnagj tensor 
                transforms.ToTensor()
            ]
        )

        # Thiết lập phương thức len sẽ trả về độ dài của dnah sách path 
        def __len__(self):
            return len(self.paths)

        # Và phương thức getitem để thực hiện lấy các phần tử 
        def __getitem__(self, index):
            # gán chỉ số cho index cho dnah sách paths để lấy ra hình ảnh 
            # tương ứng
            path = self.paths[index]
            # sử dụng hàm Image.open để đọc hình ảnh anyf 
            img = Image.open(path)
            # sau đó áp dụng biến đổi lên hình ảnh 
            return self.transform(img)
        

# Thiết lập phương thức sử dụng để tải các hình ảnh từ thư mục 
# và áp dụng lên nó một trình tự để có thể thực hiện các công việc biến đổi 
# đến hết dnah sách các hình ảnh 
# cuối cùng sẽ trả về dnah sách hình ảnh và các hình ảnh được tăng cường theo nó 
def get_image_dataloader(
        folder,
    *,
    batch_size,
    image_size,
    shuffle = True,
    cycle_dl = True,
    pin_memory = True
):
    # thực hiện tải accs hình ảnh từ thư mục 
    ds = Dataset(folder, image_size)
    # sau đó thực hiện phân chia và xáo chộn danh sách các hình ảnh 
    dl = data.DataLoader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory)

    if cycle_dl: # tham số này = True 
        # áp dụng lên dl một chu trình nó sẽ thực hiện phân chi và xáo chộn liên 
        # tục các lô trong dl cho đến khi hết danh sách này 
        dl = cycle(dl)
    return dl