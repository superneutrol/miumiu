import json 
from pathlib import Path 

import numpy as np 
import torch 
from PIL import Image 


from panopticapi.utils import rgb2id
from Utils.box_ops import masks_to_boxes

from .coco import make_coco_transforms
import transforms as T 

# Thiết lập lớp phương thức CocoPanoptic 
# Sử lý các vấn đề dữ liệu coco . Dữ liệu json này thường được chứa các trường như images
# :Images, Annotations và categorie mỗi trường chứa thông tin cụ thể . 
# 1: Images một dnah sách các hình ảnh, mỗi hình ảnh có id, file_name và thông tin khác
# 2: Annotations : Một danh sách các annotations, mỗi annotations có id, image_id, category_id
# và thông tin về segmentation. 
# 3: categories: Một danh sách các loại đối tượng (categories) có thể có trong hình ảnh, 
# mỗi categories có id và name
class CocoPanoptic: 
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        # Mở file ann_file ở chế độ dữ liệu đọc và tải dữ liệu JSON vào biến self.coco
        with open(ann_file, 'r') as f: 
            self.coco = json.load(f)

            # sắp xếp danh scahs các hinh ảnh trong trường 'images' theo thứ tự 
            # tăng dần của id 
            self.coco['images'] = sorted(self.coco['images'], key =lambda x: x['id'])
            # kieemr tra tinhs nhaats quans giữa images và annotations 
            if 'annotations' in self.coco: 
                # lấy ra hình ảnh và chú thích của hình ảnh 
                for img, ann in zip(self.coco['images'], self.coco['annotations']):
                    #đảm bảo rằng tên file của hình ảnh và tên của file annotation 
                    # trùng khớp với nhau (không tính phần mở rộng)
                    assert img['file_name'][:-3] == ann['file_name'][:-4]
                    # lưu các tham số vào các biến instance của lớp 
                    self.img_folder = img_folder
                    self.ann_folder = ann_folder
                    self.ann_file = ann_file 
                    # bao gồm cả phép biến đổi transformers 
                    self.transforms = transforms
                    # và mặt nạ masks 
                    self.return_masks = return_masks


    # Thiết lập phương thức __getitem có chức năng cho phép lớp CocoPanoptic hoạt động 
    # như một dataset trong Pytorch, nơi mà mỗi phần tử có thể được truy cập thông qua chỉ số 
    # khi phương thức này được gọi, nó sẽ trả về hình ảnh và thông tin liên quan đến annotation
    # bao gồm masks và labels cho mỗi segment trong hình ảnh . Điều này hữu ích cho việc huấn luyện 
    # các mô hình học máy với dữ liệu panoptic 
    def __getitem__(self, idx):
        # kiểm tra xem annotations có nằm trong danh scahs côc hay không 
        # coco là một biến instance của lớp CocoPanoptci được sử dụng để lưu trữ 
        # dữ liệu từ file Json chứa thông tin về hình ảnh và annotations (chú thích) cho bộ dữ
        # liệu Coco . 
        if "annotations" in self.coco: 
            #lấy ra thông tin của annotations từ trường self.coco theo chỉ số idx
            ann_info = self.coco['annotations'][idx]
        # trường hợp annotations không tồn tại trong trường coco
        else: 
            # gán ann_info = thông tin hình ảnh 
            ann_info = self.coco['images'][idx]
        # tạo đường dẫn đến file ảnh 
        # đường dẫn path có định dạng 'folder_image_name/ann_infor_name.jpg'
        img_path = Path(self.img_folder) / ann_info['file_name'].replace(',png','.jpg')
        # và tạo một đường dẫn file chú thích đường dẫn có định dạng 
        # 'folder_name/ann_infor_name' 
        ann_path = Path(self.ann_folder) / ann_info['file_name']

        # mở các ảnh trong đường dẫn và chuyển ảnh về dạng ảnh RGB 
        img = Image.open(img_path).convert('RGB')
        # lấy ra kích thước 2 chiều của hình ảnh h , w
        w, h = img.size 
        # kiểm tra xem thông tin của các phân đoạn segment_id có nằm trong 
        # danh sách thông tin ann_infor [annotation là danh sách cac ann mỗi ann gồm id , image_id
        # id_categori, segment_infor]
        if "segment_info" in ann_info: 
            # # Đọc file annotation và chuyển đổi nó thành một mảng numpy với kiểu dữ liệu uint32
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32) 
            # chuyển đổi mày RGB sang id tương ứng để tạo mask cho mỗi segment 
            # rgb2id là một hàm được sử dụng để chuyển đổi màu RGB của mỗi pixels trong mask annotation
            # thành một id duy nhất cho mỗi segment . 
            mask = rbg2id(masks)

            # duyệt qua danh sách segment info trong danh scahs annotation
            # lấy ra thông tin các phân đoạn
            for ann in ann_info['segment_info']:
                # sau đó lấy ra ID của mỗi phân đoạn tức id của mỗi segment 
                ids = np.array([ann['id']])
            # tạo masks cho mỗi segment bằng cách so sáng ID của segmetn với mảng masks 
            masks = masks == ids[: , None, None]

            # # Chuyển đổi masks thành tensor pytorch với kiểu dữ liệu uint8
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            
            # Tạo tensor chứa các category_id từ thông tin 'segments_info'
            labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        # tạo một từ điển target để lưu trữ các giá trị  như mask , labels , boxes .. 
        # của các hình ảnh 
        target = {}
        # kiểm tra điều kiện xem id của ảnh "image_id" có thuộc danh sách ann_info
        if "image_id" in ann_info: 
            # gán cho từ điển 1 căpk key , values 
            # với key = image_id và giá trị = image_id trong danh sách annotations
            target['image_id'] = torch.tensor([ann_info['image_id']])
        # còn không 
        else: 
            # lấy id của hình ảnh gán cho target['image_id']
            target['image_id'] = ann_info['id']
        
        # nếu như có tồn tại giá trị return máks 
        if self.return_masks: 
            # gán cho từ điển 1 cặp key , values với key = masks và values = tensor mask 
            target['mask'] = masks 
        # sau đó thêm vào từ điển target labels 
        target['labels'] = labels 
        # thực hiện tạo tọa độ hộp giới bạn từ mặt nạ mask 
        # kêy gọi hàm masks_to_boxes nhận đầu vào là mask shape =[N , h , w]
        target['boxes'] = masks_to_boxes(mask) # trả về tọa độ trên trái và dưới phải của hộp giưới hạn 
        # thêm vao từ điển size = h , w
        target['size'] = torch.as_tensor([int(h), int(w)])
        # và kích thước nguyên bản cho hình ảnh  h , w
        target['orig_size'] = torch.as_tensor([int(h), int(w)])

        # kiểm tra xem segment_info có trong dnah sách ann_info(annotations)
        if "segment_info" in ann_info: 
            # duyệt qua 1 list gồm 2 chuõi 'iscrowd' và 'area' 
            for name in ['iscrowd', 'area']: # 
                # thêm vào từ điển target 1 giá trị name là tên của thông tin đoạn 
                target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        # kiểm tra xem chế độ chuyển đổi có tồn tại != None 
        if self.transforms is not None: 
            # thực hiện phép biến đổi hình ảnh 
            img , target = self.transforms(img, target)

        # trả về các hình ảnh và từ điển target 
        return img , target 

    # thiết lập pt __len__ lấy độ dài danh sách ảnh từ trường coco 
    def __len__(self):
        return len(self.coco['images'])
    
    # và phương thức lấy ra chiều của hình ảnh h , w
    def get_height_and_width(self, idx):
        # lấy ra hình ảnh từ trường cooco theo idx tương ứng 
        img_info = self.coco['images'][idx]
        # lấy ra chiều cao và chiều rộng của hình ahr 
        height = img_info['height']
        # từ danh sách img_info
        width = img_info['width']
        # cuối cùng trả về chiều cao và chiều rộng của hình ảnh đươc lấy theo chỉ số idx
        return height, width


# xây dựng dữ liệu dataset coco
# nhận đầu vào gồm 1: Image_set một chuỗi chỉ định tập dữ liệu cần xây dựng , có thể là train hoặc val
# 2: args :Một đối tượng chứa các tham số cần thiết, bao gồm đường dẫn đến thư mục 
# hình ảnh và annotations .
def build(image_set, args):
    # tạo 2 đường dẫn thư mục là img_fol và ann_fol
    img_folder_root = Path(args.coco_path)
    ann_folder_root = Path(args.coco_panoptic_path)
    # kiẻm tra và đảm bảo rằng 2 thư mục trên đã được tạo 
    assert img_folder_root.exists(), f'provided COCO path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided COCO path {ann_folder_root} does not exist'
    # đặt tên mode = 'panoptic'
    mode = 'panoptic'
    # tạo một từ điển path gồm data_train , data_test
    # là đường dẫn đến các file annotations cho tập train và validation 
    PATHS = {
        "train": ("train2017", Path("annotations") / f'{mode}_train2017.json'),
        "val": ("val2017", Path("annotations") / f'{mode}_val2017.json'),
    }

    # lấy đường dẫn đến thư mục hình ảnh và file annotations dựa trên tập dữ liệu 
    # (train hoặc val)
    img_folder, ann_file = PATHS[image_set]
    # TẠO ĐƯỜNG DẪN ĐẦY ĐỦ ĐÉN THƯ MỤC HÌNH ẢNH BẰNG CÁCH NỐI ĐƯỜNG DẪN GỐC VỚI TÊN THƯ MỤC
    img_folder_path = img_folder_root / img_folder
    # TẠO ĐƯỜNG DẪN ĐẾN THƯ MỤC CHỨA MỤC ANNOTATIONS SỬ DỤNG ĐỊNH DNAGJ ĐƯỢC ĐẶT TÊN TRƯỚC
    ann_folder = ann_folder_root / f'{mode}_{img_folder}'
    # TẠO MỘT ĐƯỜNG DẪN ĐẦY ĐỦ ĐẾN ANNOTATION JSON . 
    ann_file = ann_folder_root / ann_file

    # tạo một đối tượng dataset bằng cacchs khởi tạo lớp CocoPanoptic với các 
    # tham số tương ứng . 
    dataset = CocoPanoptic(img_folder_path, ann_folder, ann_file,
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)
    # trả về đối tượng dataset 
    return dataset