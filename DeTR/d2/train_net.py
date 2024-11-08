""""
DETR Training Script.

This script is a simplified virsion of the training script in detectron2/tools.

"""
import os 
import sys 
import itertools 

# Trèn một chuỗi vào 1 đường dẫn có sẵn trước vị trí được chỉ định 
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import time 
import typing 
from typing import Any, Dict, List, Set

import torch 

import detectron2.utils.comm as comm 
from d2.detr import DetrDatasetMapper, add_detr_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from detectron2.solver.build import maybe_add_gradient_clipping


# Thiết lập một lớp Trainer có sẵn được gọi là DefaultTrainer để thích ứng với DETR 
# một kiến trúc mô hình dành cho phát hiện đối tượng 
# Lớp Trainer kế thừa DefaultTrainer 
class Trainer(DefaultTrainer):
    """
    
    Extention of the Trainer class adapted to DETR.

    """

    # Xây dựng một phương thức lớp với @classmethod 
    #  Điều này có nghĩa là phương thức đó không làm việc với một instance cụ thể của lớp, mà làm việc với chính lớp đó.
    @classmethod 
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
            Create evalutor(s) for a give dataset.
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """ 
        # kieemr tra xem nếu như output_folder = None 
        if output_folder is None: 
            # gán cho output_folder bằng một đừng dẫn được nối bằng 
            # cfg.OUT_DIR với inference 
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        # trả về bộ dữ liệu Coco Eval với output_folder đã đữoc xử lý 
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    # Tương tự như trên xây dựng một phuuwong thức lớp 
    # có tên là build_train_loader 
    @classmethod 
    def build_train_loader(cls, cfg):
        # nếu kiến trúc mô hình đang xử lý là DETR 
        if "Detr" == cfg.MODEL.META_ARCHITECTURE:
            # áp dụng DatasetMapper cho cfg
            mapper = DetrDatasetMapper(cfg, True)
        # nếu không phải 
        else:
            # gán cho mapper = None 
            mapper = None
        # trả về kết quả của dữ liệu chuẩn bị cho quá trình training detection
        return build_detection_train_loader(cfg, mapper=mapper)
    
    # xây dựng trình tối ưu hóa cho model 
    @classmethod 
    def build_optimizer(cls, cfg, model):
        # khởi tapk params là một danh sách , trong danh sách này chứa 
        # dữ liệu dạng từ điển 
        params: List[Dict[str: Any]] = []
        # tương tự với một tập set (là loại dữ liệu không có thứ tự và không được lập chỉ mục)
        # mỗi phần tử trong set là không thể thay đổi và duy nhất 
        # chứa các tham số của mô hình 
        memo: Set[torch.nn.parameter.Parameter] = set()
        # duyệt qua danh sách chứa tên và tham số của nó trong dnah sách chứa tham số mô hình 
        for key, value in model.named_parameters(recurse=True):
            # nếu như các giá trị được tính toán qua gradient 
            if not value.requires_grad: 
                continue 
            # Tránh trùng lặp thông số 
            # kiểm tra các dữ liệu của từ điển named_parameter có nằm trong 
            # tập set(nemo)
            if value in memo: 
                continue 
            # thêm value vào set memo 
            memo.add(value)
            # lấy ra leaning_rate được cấu hình gán nó cho lr 
            lr = cfg.SOLVER.WEIGHT_DECAY
            # và trọng số phân giã 
            weight_decay = cfg.SOLVER.WEIGHT_DECAY 
            # NẾU NHƯ CÓ BẤT KỲ KEY NÀO = BACKBONE 
            if "backbone" in key: 
                # nhân lr với một hệ số của backbone 
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            # thêm vào danh sách params một từ điển gồm params: value ; lr: lr ; weight_decay: weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        # Thiết lập phương thức cắt gradient đầy đủ 
        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            # lấy giá trị cắt tỉa từ cấu hình tham số 
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            # tạo một biến boolean xác định xem tính năng xắt tỉa gradient toàn mô hình có được kích 
            # hoạt hay không, dựa trên 3 điều kiện
            enable = (
                # cắt tỉa gradient phải được kich hoạt
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                # kiểu cắt tỉa full_model 
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                # và giá trị cắt tỉa phải lớn hơn không 
                and clip_norm_val > 0.0
            )

            # tạo một lớp của optimzer nếu như enable = True . 
            # lớp này ghi đè phuuwong thức step để thưucj hiện cát tỉa gradient 
            # trước khi gọi phương tức step của lớp cha 
            class FullModelGradientClippingOptimizer(optim):
                # phưuong thưc step sẽ đựoc gọi khi cập nhật cac tham số  của mô hình 
                def step(self, closure=None):
                    # tạo một interator cho tất cả các tham số của mô hình all_params 
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    # áp dụng cắt tỉa gradient(clip_grad_norm) cho tất cả các tham số 
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    # gọi lại phuuwong thức step của lớp cha để cập nhật tham số 
                    super().step(closure=closure)
            # trả về lớp FullModelGradient... nếu tính năng cắt tỉa được kích hoạt 
            return FullModelGradientClippingOptimizer if enable else optim
        

        # lấy ra kiểu tối ưu hóa từ cấu hình 
        optimizer_type = cfg.SOLVER.OPTIMIZER 
        # nếu như trình tối ưu hóa là SCHOCHATICH GRADIENT DESCENT 
        if optimizer_type == "SGD":
            # áp dụng cắt gradient cho trình tối ưu hóa để ngăn chặn vấn đề độ dốc quá lớn 
            # tránh overfiting, tăng cường khả năng tổng quát hóa ....
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        # Nếu như trình tối ưu hóa được sử dụng là Adam 
        elif optimizer_type == "ADAMW":
            # ta áp dụng cắt tỉa gradient và đưa vào nó các tham số cần thiết 
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            # nếu không thỏa mãn 2 trình tối ưu hóa trên 
            # đưa ra một lỗi và dừng trương trình 
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        # nếu như không phải cấu hình clip_type là full_model 
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            # gán cho optimizer = phép cắt tỉa gradient cho optimizer dưah trên tham số cấu hình 
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        # trả về kết quả cuối cùng 
        return optimizer



# Thiết lập phương thức khỏi tạo cácc cấu hình và cài đặc 
# các biểu diễn cơ sở 
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # cấu hình cho detr cfg
    add_detr_config(cfg)
    # hợp nhất các file config 
    cfg.merge_from_file(args.config_file)
    # hợp nhất danh sách các tham số optm
    cfg.merge_from_list(args.opts)
    # thực hiện đóng bắng tham số 
    cfg.freeze()
    # cài đặt một định nghĩa cho cfg với tham số args 
    default_setup(cfg, args)
    return cfg


# Thiết lập phương thức main cho trương trình 
def main(args):
    # Thiết đặt tham số được truyền vào 
    cfg = setup(args)

    # nếu như các tham số chỉ được sử dụng để xác thực 
    if args.eval_only: 
        model = Trainer.build_model(cfg)
        # Load trọng số của mô hình đã được huấn luyện trước đó cho nhiệm 
        # vụ xác thực dữ liệu 
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        # train model 
        res = Trainer.test(cfg, model)
        # nếu như đang sử dụng quá trình sử lý chính 
        if comm.is_main_process():
            # xác minh kết quả thông qua cfg và tham số đầu ra res của model được 
            # tái huấn luyện dựa vào tham số của mô hình ban đầu 
            verify_results(cfg, res)

        return res
    
    # tải các tham số và phân bổ cho quá trình 
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()



#  kiểm tra xem module name có được chạy trực tiếp hay không 
# và thực hiện các hành động cụ thể 
# kiểm tra xem tên module hiện tại có phải là __main__ hay không 
if __name__ == "__main__":
    # gọi một hàm default_argument_parser() để phân tích các argument dòng lệnh 
    # được cung cấp khi chạy script
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # hàm lunch được gọi với các tham số cần thiết. Hàm này có thể được sử dụng 
    # để khởi chạy một chương trình với accs cấu hình phân tán, nơi main là hàm 
    # chính cần được thực thi, 
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        # thứ tự của máy tính hiện tại, 
        machine_rank=args.machine_rank,
        # dist_url là URL dùng để thiết lập kết nối phân tán,
        dist_url=args.dist_url,
        #  và args là một tuple chứa đối tượng args với các argument dòng lệnh.
        args=(args,),
    )
