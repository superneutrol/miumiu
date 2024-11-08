import torch.utils.data 
import torchvision 
from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    # sử dụng vòng lặp for với 10 lần lặp 
    for _ in range(10):
        # nếu như dataset là một instace của torch.utils.data.Subset 
        # thì lấy dataset gốc từ nó
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    
    # nếu dataset là một instance của torchvision.datasets.CocoDetection
    # trả về đối tượng CôC API liên kết với nó . 
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        # trả về dataset coco 
        return dataset.coco 
    # hàm isinstance trong python được sử dụng để kiểm tra xem một đối tượng 
    # có phải là instance của 1 lớp cụ thể hay không . Hoặc là một trường hợp của 1
    # lớp con của lớp đó. 

# Thiết lập phương thức BUILd_dataset để xây dựng nguồn tài nguyên coco
def build_dataset(image_set, args):
    # kiểm tra xem tham số args có phải là coco tức là tập dữ liệu 
    if args.dataset_file == 'coco': 
        # nếu là coco thực hiện việc build_coco data
        return build_coco(image_set, args)
    # kiểm tra xem data_file có phải là coco_panopticapi hay không 
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        # để tránh làm panoptic cần thiết cho cocco
        from .coco_panoptic import build as build_coco_panoptic
        # trả về kết quả coco panoptic 
        return build_coco_panoptic(image_set, args)
    # nếu không thỏa mãn 2 điều kiện trên ,ném ra một cảnh báo 
    raise ValueError(f'dataset {args.dataset_file} not supported')
