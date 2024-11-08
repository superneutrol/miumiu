import torch 
import clossalat 
from clossalai.core import global_context as gpc
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.logging import disable_existing_loggers, get_dist_logger

import wandb 
from config.config import CFG 
from build_datalooader import build_dataloader 
from lamda_pytorch import lamda_model 
from Utils.utils import LaMDA_Loss, AutoregressiveWrapper 

from transformers import AutoTokenizer 

# Thiết lập phương thức LamDA_Train cho mô hình ngôn ngữ LaMDA 
def LaMDA_Trainer(cfg: CFG):
    # đảm bảo rằng GPU được trong trạng thái sẵn sàng 
    assert torch.cuda.is_available()
    # và vô hiệu hóa một trình ghi nhật ký loggers 
    # sử fungj hàm disable_existing_loggers(), logging rất quan trọng để gỡ lỗi và theo 
    # dõi việc thực thi mã 
    disable_existing_loggers()

    # khởi tạo một trình phân tích cú pháp
    parser = colossalai.get_default_parser()

    # sau đó sử dụng một đối số dòng lệnh --user_trainer với thông điệp trợ giúp 
    # "whether to use trainer."
    # action = 'store_true' chỉ ra rằng nếu đối số được cung cấp, nó sẽ được thiết lập thành True 
    parser.add_argument(
        '--use_trainer', 
        action='store_true',
        help='whether to use trainer'
    )

    # phân tích cú pháp đối số dòng lệnh. 
    # parser.parse_args() phân tích cú pháp các đối số dòng lệnh được truyền vào script 
    # các đối số được phân tích cú pháp được lưu trong biến args 
    args = parser.parse_args()

    # kiểm tra xem cfg.use_zero có bằng True hay không 
    if cfg.use_zero == True: 
        # nếu đúng, nó không lam gì cả (pass)
        pass 
    else: # nếu không gọi cấu hình được chỉ định 
        colossalai.lunch_from_torch(
            config='./lamda_pytorch/config/colossal_config.py', 
            seed = cfg.seed
        )

    # đảm bảo rằng một cấu hình config có tồn tại 
    assert hasattr(gpc.config, "EPOCHS"), "Please provide NUM_EPOCHS in your configuration"

    # Colossal logger trình ghi nhật ký khổng lồ 
    # tức là một logger phân tán từ thư viện ColossalAI có thể được sử dụng để 
    # ghi lại các thông tin quan trọng trong quá trình huấn luyện 
    logger = get_dist_logger()
    # ghi vào nhật ký một thông tin  chỉ ra rằng thông điệp này chỉ được ghi lại bởi 
    # quá trình có rank 0 trong một môi trường huấn luyện phân tán 
    logger.info("initialized evironment", ranks=[0])

    # Thiết lập mô hình LaMDA 
    model = lamda_model()
    # bọc mô hình LaMDA trong một lớp AutogressiveWrapper điều anyf có thể biến mô hình 
    # thành một mô hình tự hồi quy, nơi đầu ra của một bước thời gian được sử dụng 
    # làm đầu vào cho bước thời gian tiếp theo 
    model = AutoregressiveWrapper(model)

    # cài đặt DataLoaders 
    if cfg.use_huggingface == True: 
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        # xây dựng bộ dữ liệu cho mô hình gòm train và eval dataset 
        train_dataloader, eval_dataloader = build_dataloaders(cfg, tokenizer)
        
        
     # optimizer function

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = gpc.config.LEARNING_RATE,
        weight_decay=gpc.config.WEIGHT_DECAY
    )


    # khởi tạo mô hình , trình tối ưu hóa , tiêu trí avf tải dữ liệu 
    engine, train_dataloader, _, _ = colossalai.initialize(
        model, optimizer, loss_fn, train_dataloader = train_dataloader 
    )
    # xử lý các batch dữ liệu 
    def batch_data_process_func(batch_data):
        data = batch_data["input_ids"]
        labels = batch_data["labels"]
        return data, labels
    
    # Thiết lập hàm xử lý dữ liệu cho các lô batch dữ liệu 
    engine.schedule.batch_data_process_func = batch_data_process_func 

    # kiểm tra nếu cấu hình sử dụng wandb 
    if cfg.use_wandb == True: 
        # khởi tạo weight và Biass logging 
        wandb.init(project = cfg.project_name)

        # chuyển engine  sang chế độ huấn luyện 
        engine.train()
        for step, batch in enumerate(train_dataloader):
            # lấy dữ liệu đầu và nhãn, chuyển chúng lên GPU 
            inputs, labels = batch['inputs'].cuda(), batch['labels'].cuda()

            # xóa gradeint cũ 
            engine.zero_grad()
            # tính toán đầu ra từ mô hình 
            outputs = engine(inputs)

            #Tính toán mất mát huấn luyện vào wandb 
            train_loss = engine.loss_fn(outputs, labels)
            # ghi lại mất mát huấn luyện vào wandb 
            wandb.log({"train_loss": train_loss})

            # Lan truyền ngược Gradient 
            engine.backward(train_loss)
            # cập nhật trọng số mô hình 
            engine.step()
            # Ghi lại bước huấn luyện vào wandb 
            wandb.log({"step": step})

            # chuyển engine sang chế độ đánh giá 
            engine.eval()
            for step, batch in enumerate(eval_dataloader):
                inputs, labels = batch['inputs'].cuda(), batch['labels'].cuda()

                # Tính toán đầu ra mô hình mà không cần cập nhật gradient 
                with torch.no_grad():
                    outputs = engine(inputs)
                    test_loss = engine.loss_fn(outputs, labels)

                    # ghi lại mất mất vào wandb 
                    wandb.log({"test_loss": test_loss})
                
                # Lan truyền  ngược gradient (có vẻ như đây là một lỗi, không nên có trong khối eval)
                engine.backward(test_loss)
                engine.step()
            
        # gửi thông báo khi huấn luyện 
        wandb.alert(
            title = 'Training Complete',
            text = "Training complete."
        )

    else:

        # Khởi tạo bộ đếm thời gian với ColossalAI
        timer = MultiTimer()

        # khởi tạo trainer với engine, bộ đếm thời gian và logger 
        trainer = Trainer(
            engine = engine,
            timer =  timer,
            logger = logger
        )

        # Danh sách các hook sẽ được sử dụng trong quá trình huấn luyện 
        # Danh sách các hook sẽ được sử dụng trong quá trình huấn luyện
        hook_list = [
            hooks.LogMetricByStepHook(),
            hooks.LossHook(),
            hooks.LogMetricByEpochHook(logger)
        ]

        # Bắt đầu quá trình huấn luyện với các thông số đã được định nghĩa
        trainer.fit(
            train_dataloader = train_dataloader,
            epochs = gpc.config.EPOCHS,
            hooks = hook_list,
            display_progress = True
        )


if __name__ == "__main__":

    cfg = CFG()

    LaMDA_Trainer(cfg)