import copy 
import logging 
import numpy as np 
import torch 

from detectron2.data import detection_utils as utils 
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen

__all__ = ["DetrDatasetMapper"]


# Thiết lập phương thức biến đổi gen áp dụng cho việc thực hiện 
# biến đổi dữ liệu 

def build_transform_gen(cfg, is_train):
    """"
    Create a list of: class: 'TransformGen' from config 
    Returns: 
        list[TransformGen]
    """
    # Kiểm tra xem is train có tồn tại 
    if is_train: 
        # GÁN CÁC THAM SỐ THEO CONFIG PARAMETERS 
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    # Nếu is_train = False 
    else:
        # gán min_size, max_size và sample_tyle = "choice"
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    # Nếu như sample_style = range 
    if sample_style == "range":
        # và đảm bảo rằng min_size == 2 
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    # ghi vào nhật ký một logging infor = list [___name__]
    logger = logging.getLogger(__name__)
    # tạo một danh sách transformGen để lưu trữ các phéo biến đổi 
    tfm_gens = []
    # nếu như is_train = True 
    if is_train:
        # thêm vào danh sách một phép biến đổi RandomFlip 
        tfm_gens.append(T.RandomFlip())
    # sau đó thêm vào danh sách một phép biến dổi resize với kích thuớc 
    # được phụ thuộc vào các tham số được truyền vào 
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    # sau đó ghi một nhật ký vào logger 
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    # trả về dnah sách các phép biến đổi 
    return tfm_gens


# Thiết lập lớp phương thức DetrDatasetMapper để thực hiện áp dụng 
# phép xử lý chuẩn bị hình ảnh và chú thích annotations tương ứng dưới dạng tensor 
class DetrDatasetMapper: 
    """
    A callable which takes a dataset dict in Detectron2 Dataset format, and map it into a format used by DETR. 

    The callable currently does the following: 
        1. Read the image from "file_name"
        2. Applies geometric transforms to the image and annotation
        3. Find and applies suitable cropping to the image and annotation
        4. Prepare image and annotation to Tensors
    """
    # Thiết lập phương thức khởi tạo 
    def __init__(self,cfg, is_train=True):
        # nếu như cfg.INPUT.CROP.ENABLE và is_train = True 
        if cfg.INPUT.CROP.ENABLE and is_train: 
            # khởi tạo một dnah sách cop_gen = 
            # 2 phép biến đổi hình ảnh là ResizeShortestEdg và RandomCrop 
            self.crop_gen = [
                T.ResizeShortestEdg([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            ]
        # Trường hợp còn lại 
        else: 
            # gán cho crop_gen  = None 
            self.crop_gen = None

        # khởi tạo biến mask_on gán nó = kết quả của mask_on từ config của Detr
        self.mask_on = cfg.MODEL.MASK_ON
        # Xây dựng biến đổi gen và truyền vào accs tham số cfg , is_train
        self.tfm_gens = build_transform_gen(cfg, is_train)
        # ghi vào nhật ký logg một Infor tên là __Name__ có thông tin theo định dạng 
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        # lấy ra định dạng của hình ảnh từ Input.Format
        self.img_format = cfg.INPUT.FORMAT
        # gán lại kết quả cho is_train 
        self.is_train = is_train

    # Thiết lập phưuowng thức call để thực hiện xây dựng các bước xử lý lớp 
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # sao chép sâu từ điển đầu vào 
        dataset_dict = copy.deepcopy(dataset_dict) # nó sẽ được thay đổi bởi mã bên dưới 
        # đọc các hình ảnh từ file_name trong từ điển với các hình ảnh được định dnagj theo 
        # kích thước img_format 
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # sử dụng phương thức check_image_size để kiểm tra kích thước của 
        # hình ảnh trong danh sách image được trích xuất với kích thước trong từ điển 
        utils.check_image_size(dataset_dict, image)

        # kiểm tra xem danh sách crop_gen = NONE 
        if self.crop_gen is None: 
            # nếu như danh sách này  = None 
            # áp dụng phép biến đổi gens cho hình ảnh 
            image , transforms = T.apply_transform_gens(self.tfm_gens, image)
        # trường hợp không = None 
        else: 
            # áp dụng phép biến đổi không trực tiếp từ tham số khởi tạo 
            image , transforms = T.apply_transform_gens(
                # lấy ra giá trị đầu tiên  của tfm_gens 
                # và lấy ra giá trị cuối cùng của danh sách tfm_gens cộng chúng với crop_gen 
                 self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
            )

        # lấy ra kích thước của hình ảnh là 2 kích thước đầu trong danh sách imgae
        image_shape = image.shape[:2]

        # tạo một khóa image với key = 1 tensor shape [C, h, w] được lưu trữ trong từ điển dataset_dict 
        # sử dụng hàm np.ascontiguousarray để đảm bảo dữ liệu được lưu trữ liên tục trong bộ nhớ
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # kiểm tra xem is_train = True tức là quá trình hiện tại có phải quá trình huan luyện
        if not self.is_train: 
            # xóa từ điển một chuỗi annotations = key values = none
            dataset_dict.pop("annotations", None)
            # trả về từ điển này 

        # kiểm tra xem nếu như annotations có trong từ điển  
        if "annotations" in dataset_dict: 
            # duyệtq au 1 dnah sách annotaions gán cho ann 
            for anno in dataset_dict["annotations"]:
                # nếu không tồn tại mask 
                if not self.mask_on:
                    # xóa các segmentations và keypoints khỏi từ điển 
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # áp dụng các phép biến đổi hình ảnh 
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                # duyệt qua dnah sách các giá trị annotations bị bỏ đi trong từ điển (được lấy ra) 
                for obj in dataset_dict.pop("annotations")
                #nếu như obj tồn tại giá trị iscrow so snahs nó nếu iscrow = 0 thì sẽ được sử lý tiếp 
                if obj.get("iscrowd", 0) == 0
            ]
            # Hàm utils.annotations_to_instances() chuyển đổi các chú thích đã biến đổi thành các đối tượng instance.
            # Các instance này đại diện cho các đối tượng hoặc vùng quan tâm riêng lẻ trong ảnh.
            instances = utils.annotations_to_instances(annos, image_shape)
            # Hàm utils.filter_empty_instances() loại bỏ bất kỳ instance trống nào (ví dụ: instance không có dữ liệu hợp lệ).
            # Các instance đã lọc được lưu lại trong dataset_dict["instances"].
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        # cuối cùng trả về từ điển dataset dict 
        return dataset_dict



