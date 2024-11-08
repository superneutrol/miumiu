""""
    CoCo Dataset which returns image_id for evaluate. 

"""

from pathlib import Path 
import torch 
import torch.utils.data 
import torchvision 
from pycocotools import mask as coco_mask 
import transforms as T 

# Thiết lập lớp phương thức CocoDetection sủ dụng để 
# tải và biến đổi dữ liệu từ bộ dữ liệu COCO cho nhiệm vụ nhận dạng hình ảnh
# VỚI COCO DÊTCTION DATA CHỨA CAC THAM SỐ NHƯ SAU , 
# IMAGE , ANNOTATIONS 
# 2: ANNOTATION SẼ CÓ 
    # SEGMENTATION LÀ THÔNG TIN MÔ TẢ HÌNH DNAGJ CỦA 1 ĐỐI TƯỢNG TRONG HÌNH ẢNH . TRONG ĐỊNH DNAGJ COCO
            # SEGMENTATION CÓ THỂ LÀ MỘT DANH SÁCH CÁC ĐIỂM CỦA ĐA GIÁC (POLYSGON) HOẶC MỘT MASK NHỊ PHÂN. 
            # THÔNG TIN NÀY SẼ ĐƯỢC SỬ DỤNG ĐỂ TẠO RA MẶT NẠ MASK BAO QUANH ĐỐI TƯỢNG . 
# CATEGORY_ID LÀ MỘT SỐ NGUYÊN ĐẠI DIỆN CHO MỘT LOẠI ĐỐI TƯỢNG ĐƯỢC PHÁT HIỆN TRONG HÌNH ẢNH 
# MỖI LOẠI ĐỐI TƯỢNG NHƯ (NGƯỜI , XE HƠI...) SẼ CÓ 1 ID DUY NHẤT. 
class CocoDetection(torchvision.datasets.CocoDetection):
    # thiết lập phương thức khởi tạo nhận đầu vào gômg các tham số 
    # image_folder , ann_file , phép biến đổi hình ảnh transforms, và mặt nạ return masks
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        # Gọi hàm khởi tạo của lớp cha và truyền vào thư hình ảnh và file chứa thông 
        # tin annotations của hình ảnh 
        super(CocoDetection, self).__init__(img_folder, ann_file)
        # Lưu trữ các phép biến đổi sẽ được áp dụng cho dữ liệu
        self._transforms = transforms
        # chuẩn bị  hàm để chuyển đổi polygon annotations thành masks nếu cần
        # polysgon annotations trong bối cảnh của dữ liệu coco LÀ CÁC ĐA GIÁC ĐỊNH NGHĨA 
        # VÙNG CHỨA ĐỐI TƯỢNG TRONG MỘT HÌNH ẢNH . MỖI ĐA GIÁC ĐƯỢC TẠO TỪ  DNAH SÁCH 
        # CÁC ĐIỂM TỌA ĐỘ 
        self.prepare = ConvertCocoPolysToMask(return_masks)

    # thiết lập phương thức __getitem__
    # để lấy ác phần tử từ CocoDetection tức là lấy một mẫu dữ liệu dựa trên chỉ số 
    # idx . Nó trả về hình ảnh và thông tin liên quan đã được biến đổi 
    def __getitem__(self, idx):
        # get images and annotation info from Coco dataset by user funcion of the Class
        # Nhận các hình ảnh và thông tin annotations tù bộ dữ liệu coco bằng cách sử dụng 
        # hàm của lớp cha 
        img, target = super(CocoDetection, self).__getitem__(idx)
        # lấy ra ID của hình ảnh 
        image_id = self.ids[idx]
        # Tạo một từ điển mới là target gán cho annotations và image_id 
        target = {'image_id': image_id, 'annotations': target}
        # áp dụng prepare để chuyển đổi annotations thành mask nếu cần 
        img, target = self.prepare(img, target)
        # nếu có phép biến đổi nào được định nghĩa , áp dụng chúng cho hình ảnh và target
        if self._transforms is not None: 
            # ap dụng phép biến đổi đó cho hình ảnh và target 
            img, target = self._transforms(img, target)
            # trả về hình ảnh và thông tin được biến đổi 
            return img , target 
        

# Thiết lập phương thức chuyển đổi coco polysgon thành mask 
# polysgon trong bối cảnh dữ liệu coco là các đa giác được biểu diễn bởi các tọa độ x , y 
# của thực thể trong hình ảnh . 
def convert_coco_poly_to_mask(segmentations, height, width):
    # khởi tạo một dnah sách mask để lưu trữ kết quả là các mặt nạ được 
    # xử lý 
    masks = []
    # lặp qua mỗi polygons trong segmentationns
    # segmentations là một danh sách chứa các polygons , mỗi polygons chứa các điểm các điểm này 
    # biểu diễn 1 đa giác như đã nói . 
    for polygons in segmentations: 
        # chuyển đổi mỗi polygons thành RLE( RUN Length Encoding).
        # Rle biểu diễn thông tin của mask một cách hiệu quả để lưu trữ thông 
        # tin về các vùng liên tục của pixels
        rles = coco_mask.frPyObjects(polygons, height , width)
        # giải mã các rle trong danh sách rles thành các mask nhị phân 0, 1
        mask = coco_mask.decode(rles)
        # kiểm tra xem mask có số chiều < 3: 
        if len(mask.shape) < 3: 
            # thêm 1 chiều mới vào mask tức chiều thứ 3
            mask = mask[..., None]
        # chuyển đổi mask thành tensor 
        mask = torch.tensor(mask , dtype=torch.uint8)
        # kêys hợp các masks nếu có nhiều hơn 1 mask cho cùng một vùng
        mask = mask.any(2)
        # thêm tensor mask vào dnah sách mask 
        masks.append(mask)
    
    # kiểm tra xem mask có  = None 
    if mask:
        # nếu không sử dụng hàm stack để nối các tensor lại với nhau theo chiều 
        # đầu tiên 
        masks = torch.stack(masks, dim=0)  # Ghép các masks lại với nhau theo chiều đầu tiên.
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)  # Tạo một tensor masks rỗng nếu không có mask nào.
    # trả về danh sách masks 
        
    return masks 

# Chuyển đổi các polygons annotations trong coco thành mặt nạ masks 
# polygon là các đa giá định định nghĩa một vùng chứa đối tượng cho hình ảnh 
# được biểu diễn bởi các điểm tọa độ 
class ConvertCocoPolysToMask(object):
    # thiết lập phương thức khởi tạo và định nghĩa các tham sốn
    def __init__(self, return_masks=False):
        self.return_mask = return_masks

    # và phương thức call để xây dựng chức năng xử lý của lớp 
    def __call__(self, image, target):
        # lấy ra kích thước của hình ảnh 
        w , h = image.size 
        # lấy ra id của hình ảnh từ từ điển target 
        image_id = target['image_id']
        # chuyển đổi danh scahs các id thành tensor 
        image_id = torch.tensor([image_id])
        
        # và cũng lấy ra các annotations của hình ảnh từ bộ từ điển target 
        anno = target["annotations"]

        # duyệt qua danh sách các annotations lấy ra các đối tượng 
        for object in anno: 
            # kiểm tra điều kiện xác định nếu như trong chú thích không tồn tại 
            # iscrowd
            if 'iscrowd' not in object or object['iscowd'] == 0:
                # thêm các object [annotations] vào danh sách ann 
                anno = [object]
        
        # lấy ra hộp giới hạn từ các chú thích OBJ trong annot
        boxes = [object['bbox'] for object in anno]
        # bảo vệ việc chống lại việc không co hộp thông qua việc thay đổi kích thước 
        # sau đó reshape tensor boxes shape = [N, 4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1,4)
        # cập nhật boxes để có định dạng [x_min, y_min, x_max, y_max]
        # bằng cacchs kết hợp 2 phần tử đầu và cuối của chiều cuối cùng lại với nhau 
        boxes[:, 2:] += boxes[:, :2]
        # giới hạn giá trị của boxes trong phạm vi kích thước hình ảnh. 
        # 1 : Thực hiện giới hạn các giá trị của cạnh trái và cạnh phải của boundingbox (
        # tức là các giá trị x_min , x_max) để chúng không nhỏ hơn 0 và không lớn hơn 
        # lớn hơn chiều ngang của hình ảnh . 
        boxes[:, 0::2].clamp_(min=0, max=h) # 
        # 2: Giới hạn các giá trị cảu cạnh trên và cạnh dưới của bounding boxes (tức là các giá trị ymin , y_max)
        # để chúng không nhỏ hơn 0 và không lớn hơn chiều cao của hình ảnh h . 
        boxes[:, 1::2].clamp_(min=0, max=h)

        # lấy category_id từ annotations và chuyển thành tensor 
        # CATEGORY_ID LÀ MỘT SỐ NGUYÊN ĐẠI DIỆN CHO MỘT LOẠI ĐỐI TƯỢNG ĐƯỢC PHÁT HIỆN TRONG HÌNH ẢNH 
        # MỖI LOẠI ĐỐI TƯỢNG NHƯ (NGƯỜI , XE HƠI...) SẼ CÓ 1 ID DUY NHẤT 
        classes = [obj['category_id'] for obj in anno]
        classes = torch.tensor(classes)

        # kiểm tra xem gia trị của self.return_masks có tồn tại hay không 
        if self.return_mask: 
            # lấy ra các annotation từ segmentation trong annotations 
            # SEGMENTATION LÀ THÔNG TIN MÔ TẢ HÌNH DẠNG CỦA 1 ĐỐI TƯỢNG TRONG HÌNH ẢNH . TRONG ĐỊNH Dạng COCO
            # SEGMENTATION CÓ THỂ LÀ MỘT DANH SÁCH CÁC ĐIỂM CỦA ĐA GIÁC (POLYSGON) HOẶC MỘT MASK NHỊ PHÂN. 
            # THÔNG TIN NÀY SẼ ĐƯỢC SỬ DỤNG ĐỂ TẠO RA MẶT NẠ MASK BAO QUANH ĐỐI TƯỢNG . 
            segmentations = [obj["segmentation"] for obj in anno]
            # # sau đó chuyển đổi các polysgon từ segmentation thành các hộp giới hạn 
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # Gán cho keyponu = None  
        keypoints = None 
        # kiểm tra đièu kiệnu xem annn và kypoints có nằm trong chú thích thứ nhất ann[0]: 
        # KEYPOINTS LÀ CÁC ĐIỂM ĐẶC BIỆT TRÊN ĐỐI TƯỢNG THƯỜNG ĐƯỢC ĐƯỚCUWR DỤNG TRONG BÀI TOÁN NHẬN DẠNG 
        # TƯ THẾ CỦA CON NGƯỜI. MỖI KEYPOINT BAO GỒM BA GIÁ TRỊ TỌA ĐỘ X, TỌ ĐỘ Y VÀ GIÁ TRỊ V BIỂU THỊ 
        # SỰ HIỆN DIỆN CỦA KEYPOINTS VD (CÓ THỂ NHÌN THẤY KEYPOINT HAY KHÔNG)
        if anno and "keypoints" in anno[0]:
            # KEYPOINTS LÀ CÁC ĐIỂM ĐẶC BIỆT 
            # lấy ra các giá keypoints từ giá trị anno 
            keypoints = [obj["keypoints"] for obj in anno]
            # sau đó chuyển nó thành 1 tensor 
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            # lấy ra số lượng keypoints từ tensor keypoints theo chiều thứ nhất 
            num_keypoints = keypoints.shape[0]
            # nếu như giá trị keypoints tồn tại 
            if num_keypoints: 
                # chuyển nó thành tensor có hình dnag [n , x, y , v]
                keypoints = keypoints.view(num_keypoints, -1, 3)
                
        # TẠO CÁC MẢNG BOOLEAN 'KEEP' ĐỂ KIỂM YTRA XEM CÁC BOUNDING BOXES CÓ HỢP LỆ HAY KHÔNG 
        # MỘT BOXES HỢP LỆ KHI Y_MAX > Y_MIN , X_MAX > X_MIN
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        # lọc lại những dự liệu không hợp lệ trong danh sách boxes. 
        boxes = boxes[keep]
        # lọc giữ lại những dữ class tương ứng với các boxes hợp lệ 
        classes = classes[keep]
        # kiểm tra xem masks tồn tại không 
        if self.return_mask: 
            # nếu có tồn tại áp dụng keep để lọc masks 
            mask = masks[keep]
        
        # tương tự áp dụng với keypoints 
        # sau đó áp dụng lọc keep cho dnah sách keypoints
        if keypoints is not None:
            keypoints = keypoints[keep]

        # tạo một từ điển target để lưu trữ các danh sách thông tin 
        target = {}
        # thêm hộp giới hạn , labels vào trong từ điển target 
        target['boxes'] = boxes 
        target['labels'] = classes # nhãn của đối tượng 
        # kiểm tra xem mặt nạ có tồn tại không 
        if self.return_mask: 
            # thêm mặt nạ masks vào từ điển target 
            target['masks'] = masks 
        # và image_id vào từ điển target 
        target['image_id'] = image_id
        # nếu như keypoints không = None 
        if keypoints is not None:
            # thêm keypoints vào từ điển target 
            target['keypoints'] = keypoints 
        
        # Thực hiện một số bước để chuyển đổi sang coco API 
        # lấy ra diện tích hình ảnh hoặc có thể là diện tích hộp boxes 
        # trong annotations và biến đổi nó thành tensor
        area = torch.tensor([obj['area'] for obj in anno])
        # tương tự như iscrowd khi is crown tồn tại trong annotations 
        # nếu không gán iscrown = 0 
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # lọc các area không thỏa mãn điều kiện ra khỏi từ điển target
        target["area"] = area[keep]
        # tương tự áp dụng nó cho iscrowd trong từ điển target
        target["iscrowd"] = iscrowd[keep]

        # Thêm vào từ điển target kích thước ban đầu của hình ảnh 
        # dưới dnagj tensor biểu diễn 
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        # và kích thước size = h , w
        target["size"] = torch.as_tensor([int(h), int(w)])

        # cuối cùng trả về image và từ điển target 
        return image, target
    

# Thiết lập phương thức tạo các phép biến đổi Cocco dataset 
def make_coco_transforms(image_set):
    # gọi đến hàm compose để xử lý các trình tự biên đổi và các phép 
    # sử lý dữ liệu 
    # gọi đến phương thức normalize = T.compose để thực hiện bình thường hóa các giá trị dữ liệu 
    # thực hiện phân phối dữ liệu trên các pixel để chúng đồng nhất hơn 
    normalize = T.Compose([
        T.ToTensor(),
        # CÁC THAM SỐ PHÂN PHỐI STD VÀ MEAN 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # tạo một dnah sách scale tà danh sách được sử DỤNG ĐỂ THỰC HIỆN CHO VIỆC RISIZE HÌNH ẢNH 
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    
    # KIỂM GTRA XEM HÌNH ẢNH CÓ PHẢI LÀ HÌNH ẢNH ĐƯỢC SỬ DỤNG CHO VIỆC TRAIN
    if image_set == 'train':
        return T.Compose([
            # ÁP DỤNG LẬT HÌNH ẢNH
            T.RandomHorizontalFlip(),
            # VÀ MỘT LỚP LỰA CHỌN NGẪU NHIÊN 
            T.RandomSelect(
                # RESIZE VÀ THỰC HIỆN COMPOSE HÌNH ẢNH 
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize, # SAU ĐÓ ÁP DỤNG PHÂN PHỐI DỮ LIỆU ĐỒNG NHẤT 
        ])

    # KIỂM TRA XEM HÌNH ẢNH CÓ PHẢI LÀ DỮ LIỆU THỬ NGHIỆM
    if image_set == 'val':
        # NẾU PHẢI CHỈ SỬ DỤNG RANDOM_SIZE VÀ PHÂN PHÔI ĐỒNG NHẤT
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    # CÒN KHÔNG NÉM RA MỘT CẢNH BÁO 
    raise ValueError(f'unknown {image_set}')


# Thiết lập một hàm build để tạo một tập dữ liệu phát hiện đối tượng 
# từ bộ dữ liệu Coco được cung cấp qua tham số  args.coco_path 
def build(image_set, args):
    root = Path(args.coco_path)
    # kiểm tra xen đường dẫn có tồn tại hay không , nếu không một ngoại lệ được ném ra với 
    # thông báo lỗi 
    assert root.exists(), f'provided COCO path {root} does not exist'
    # đặt một biến mode với giá trị 'instances' thuwongf được sử dụng để chỉ định loại dữ liệu 
    # annotaion trong bộ dữ liệu coco 
    mode = 'instances'
    # tạo một từ điển path chứa đường dẫn đến các thư mục hình a và tệp annotation JSON cho tập huấn luyện (train) và tập kiểm định (val).
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset


        








