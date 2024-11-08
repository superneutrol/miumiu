import click
import torch

from torch import nn
from typing import List
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from embedding_reader import EmbeddingReader
from accelerate.utils import dataclasses as accelerate_dataclasses

from dalle2_pytorch.utils import Timer
from dalle2_pytorch.trackers import Tracker
from dalle2_pytorch import DiffusionPriorTrainer
from dalle2_pytorch.dataloaders import get_reader, make_splits
from dalle2_pytorch.train_configs import (
    DiffusionPriorConfig,
    DiffusionPriorTrainConfig,
    TrainDiffusionPriorConfig,
)


# helpers
# Định nghĩa hàm cosine ước lượng
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

# Xây dựng phương thức exits phương thức này sẽ trả về 
# chính nó nếu như nó có tồn tại 
def exists(val):
    # Đầu vào val là một tensor, list cũng có thể là một dictionary 
    return val is not None

# Xây dựng phương thức all_between
# phương thức này sẽ trả về một giá trị boolean 
# qua một phép điều kiện phương thức này nhận đầu vào một giá trị list[values] và 2 
# biến đại diện cho 2 giới hạn trên và dưới 
def all_between(values: list, lower_bound, upper_bound):
    # duyệt qua danh sách các tensor trong danh sách values
    for value in values:
        # đảm bảo value thảo mãn 2 giới hạn này 
        if value < lower_bound or value > upper_bound:
            # Nếu như điều kiện này đúng tức nó đã quá giới hạn 
            # return False
            return False

    # Còn lại retaun True 
    return True


# Xây dựng phương thức Make_model thực hiện xây dựng 
# mô hình khuếch tán của DALL.E2 
def make_model(
        # Định cấu hình các tham số cho kiến trúc prior diffusion và train diffusion
        prior_config: DiffusionPriorConfig,
        train_config: DiffusionPriorTrainConfig,
        device: str = None,
        accelerator: Accelerator = None,
):
    # Khởi tạo một từ điển cấu hình tham số cho mô hình 
    # thông qua hàm prior_config.create() kết quả gán cho biến diffusion_prior
    diffusion_prior = prior_config.create()

    # Khởi tạo mô hình trainer 
    trainer = DiffusionPriorTrainer(
        # Cấu hình trainer với các tham số cần thiết 
        diffusion_prior=diffusion_prior, 
        # learning_rate 
        lr = train_config.lr,
        # weight_decay 
        wd = train_config.wd,
        # Tham số chuẩn hóa dữ liệu cho gradient
        max_grad_norm=train_config.max_grad_norm,
        # 
        amp=train_config.amp,
        use_ema=train_config.use_ema,
        device=device,
        # Tiến trình (số lượng tiến trình sẽ tham gia)
        accelerator=accelerator,
        # Số lượng bước khởi động 
        warmup_steps=train_config.warmup_steps,
    )

    # Cuối cùng trả về mô hình trainer 
    return trainer 



# Thiết lập phương thức tracker phương thức này được sửv dụng để theo dõi quá 
# trình huấn luyện và ghi lại thông tin xảy ra trong trình đào tạo 
def create_tracker(
    accelerator: Accelerator,
    config: TrainDiffusionPriorConfig,
    config_path: str,
    # dummy: Cờ chỉ định xem chế độ giả lập (dummy) có được kích hoạt hay không (mặc định là False).
    dummy: bool = False,
) -> Tracker: 
    # Định cấu hình cho tracker 
    tracker_config = config.tracker

    # Sau đó, nó tạo một từ điển accelerator_config chứa thông tin về việc phân tán 
    # (distributed) và tối ưu hóa (mixed precision) của accelerator.
    accelerator_config = {
        "Distributed": accelerator.distributed_type
        != accelerate_dataclasses.DistributedType.NO,
        "DistributedType": accelerator.distributed_type,
        "NumProcesses": accelerator.num_processes,
        "MixedPrecision": accelerator.mixed_precision,
    }

    # Tiếp theo, mã tạo đối tượng tracker bằng cách gọi hàm tracker_config.create với các tham số tương ứng.
    tracker: Tracker = tracker_config.create(
        config, accelerator_config, dummy_mode=dummy
    )
    # Cuối cùng, nó lưu cấu hình vào tệp prior_config.json và trả về đối tượng tracker đã tạo.
    tracker.save_config(config_path, config_name="prior_config.json")

    return tracker


# Xây dựng phương thức pad_gather_reduce phương thức này thực hiện đệm 
# các dữ liệu hoặc accs tensor thông qua tất cả các tiến trình và tập trung
def pad_gather_reduce(trainer: DiffusionPriorTrainer, x, method="mean"):
    """
    pad a value or tensor across all processes and gather

    params:
        - trainer: a trainer that carries an accelerator object
        - x: a number or torch tensor to reduce
        - method: "mean", "sum", "max", "min"

    return:
        - the average tensor after maskin out 0's
        - None if the gather resulted in an empty tensor
    """
    # Đảm bảo rằng chuỗi method là một trong các phần tử thuộc danh sách 
    assert method in [
        "mean",
        "sum",
        "max",
        "min",
        # Hàm này giới hạn khả năng có thể trong [sum, mean, max, min]
    ], "This function has limited capabilities [sum, mean, max, min]"
    # Đảm bảo x là dữ liệu có định dạng 
    assert type(x) is not None, "Cannot reduce a None type object"

    # wait for everyone to arrive here before gathering
    # Nếu như kiểu dữ liệu của x không phải alf một tenssor 
    if type(x) is not torch.Tensor:
        # Ta chuyển x thành tensor sử dụng hàm torch.tensor 
        x = torch.tensor([x])

    # verify that the tensor is on the proper device
    # Xác minh rằng tensor ở trên thiết bị phù hợp 
    x = x.to(trainer.device)

    # pad across processes
    # Thực hiện đệm các tensor thông qua các quá trình xử lý 
    padded_x = trainer.accelerator.pad_across_processes(x, dim=0)

    # gather across all procesess
    # hàm gather sẽ tập trung tất cả các tensor được đêmj trên tất cả các tiến trình
    gathered_x = trainer.accelerator.gather(padded_x)

    # mask out zeros
    # Gán cho các phần được đệm trong tensor = 0 kết quả gán cho masked_x
    masked_x = gathered_x[gathered_x != 0]

    # if the tensor is empty, warn and return None
    # Kiểm tra xem các phần tử trong danh sách masked == 0 
    if len(masked_x) == 0:
        click.secho(
            f"The call to this method resulted in an empty tensor after masking out zeros. The gathered tensor was this: {gathered_x} and the original value passed was: {x}.",
            fg="red",
        )
        return None

    # NẾU METHOD == MEAN
    if method == "mean":
        # Tính trung bình tensor masked
        return torch.mean(masked_x)
    # Nếu như là sum 
    elif method == "sum":
        # Thực hiện tính tổng
        return torch.sum(masked_x)
    # Nếu là max 
    elif method == "max":
        # Tính max cho tensor masked_x
        return torch.max(masked_x)
    # Cuối cùng là min  
    elif method == "min":
        return torch.min(masked_x)



# Xây dựng phương thức save_trainer để lưu trữ lại các kết quả 
# trong quá trình training model prior diffusion 
def save_trainer(
    # Định nghĩa các cấu hình 
    tracker: Tracker,
    trainer: DiffusionPriorTrainer,
    is_latest: bool,
    is_best: bool,
    epoch: int,
    samples_seen: int,
    best_validation_loss: float,
):
    """
    Logs the model with an appropriate method depending on the tracker
    """
    # Đợi cho đến khi tất cả các tiến trình được xử lý trên GPU được hoàn tất
    trainer.accelerator.wait_for_everyone()

    # Kiểm tra xem có đang ở trên quá trình chính hay không 
    if trainer.accelerator.is_main_process:
        # sử dụng hàm click.secho để in thông điệp dòng lệnh 
        # (command line interface). Nó hoạt động tương tự như câu lênhk print 
        # trong python, nhưng khác biệt chình 
        click.secho(
            f"RANK:{trainer.accelerator.process_index} | Saving Model | Best={is_best} | Latest={is_latest}",
            fg="magenta",
        )

    # Sử dụng trình theo dõi tracker để lưu lại thông tin của mô hình tại thời điểm 
    # huấn luyện 
    tracker.save(
        trainer=trainer,
        is_best=is_best,
        is_latest=is_latest,
        epoch=int(epoch),
        samples_seen=int(samples_seen),
        best_validation_loss=best_validation_loss,
    )


# Xây dựng phương thức recall phương thức này được sử dụng để tải 
# lại tham số cho mô hình 
def recall_trainer(tracker: Tracker, trainer: DiffusionPriorTrainer):
    """
    Loads the model with an appropriate method depending on the tracker
    """

    # Kiểm tra xem có đang ở quá trình chính của mô hình hay không 
    if trainer.accelerator.is_main_process:
        click.secho(f"Loading model from {type(tracker.loader).__name__}", fg="yellow")

    # Sử dụng trình theo dõi tracker để lấy ra một từ điển trạng thái của quá trình hiện tại
    state_dict = tracker.recall()

    # hàm trainer.load được sử dụng để trích xuất các tham số 
    trainer.load(state_dict, strict=True)

    # Cuối cùng trả về các thông tin 
    return (
        # Số kỷ nguyên 
        int(state_dict.get("epoch", 0)),
        # Lấy ra Chi phí xác thực tốt nhất 
        state_dict.get("best_validation_loss", 0),
        # Và danh sách các mẫu 
        int(state_dict.get("samples_seen", 0)),
    )



# Xây dựng phương thức report_validation_loss được sử dụng để tính toán 
# lỗi xác thực trên một tập hợp con dữ liệu nhất định
def report_validation_loss(
        # Định nghĩa Trình huấn luyện cho mô hinh DiffusionPrior 
        trainer: DiffusionPriorTrainer, 
        # Data Loader 
        dataloader: DataLoader, 
        # Điều kiện hóa văn bản 
        text_conditioned: bool, 
        # ema sử dụng cho việc tính toán chi phí tổng hợp 
        use_ema: bool,
        # Trình giám sát 
        tracker: Tracker , 
        split: str,
        tracker_folder: str,
        loss_type: str,

    ):
    """
    Compute the validation loss on a given subset of data.
    """

    # Kiểm tra xem trình đào tạo có đang sử dụng quá trình chính hay không
    if trainer.accelerator.is_main_process:
        # sử dụng hàm click.secho để in ra một thông điệp dòng lệnh comand line interface
        click.secho(
            f"Measuring performance on {use_ema}-{split} split",
            # sử dụng màu nền = Green 
            fg="green",
            blink=True,
        )

        # Tạo một tensor shape = 1 để lưu trữ kết quả của loss 
        # gán cho biến total_loss 
        total_loss = torch.zeros(1, dtype=torch.float, device=trainer.device)

        # Lấy ra các nhúng hình ảnh và dữ liệu văn bản trong trình tải 
        for image_embeddings, text_data in dataloader:
            # chuyển danh sách các nhúng hình ảnh cho thiết bị 
            image_embeddings = image_embeddings.to(trainer.device)
            # Thực hiện tương tự với text_data 
            text_data = text_data.to(trainer.device)

            # Khởi tạo từ điển input_args để lưu kết quả của các image_embeddings
            input_args = dict(image_embed = image_embeddings)

            # kiểm tra xem có điều kiện hóa văn bản hay không 
            if text_conditioned:
                # nếu có lưu chữ text cùng với từ điển input_args trước đó 
                # vào từ điển input_args 
                input_args = dict (**input_args, text=text_data)
            
            # Trường hợp còn lai không có điều kiện hóa văn bản 
            else:
                # ta lưu chữ text_embed cùng với image_embed vào từ điển input_arags
                input_args = dict(**input_args, text_embed=text_data)

            # nếu như có sử dụng việc tính toán loss tổng hợp 
            if use_ema:
                # ta sử dụng hàm trainer.ema_diffusion_prior để tính toán chi phí 
                # tổng hợp thông qua từ điển input_args
                loss = trainer.ema_diffusion_prior(**input_args)
            # trường hợp còn lại 
            else:
                # Ta thực hiện tính loss thông thường cho trình huấn luyện
                loss = trainer(**input_args)

            # Cộng total_loss với giá trị loss 
            total_loss += loss

        
        # Tính toán chi phí trung bình thông qua tiến trình đang sử dụng 

        # sử dụng hàm pad_gather_reduce để đệm các tensor và thực hiện chức năng tính
        # toán tổn thất chi phí của tất cả các loss sử dụng tổn thất trung bình 
        # cho tất cả các tensor 
        avg_loss = pad_gather_reduce(trainer, total_loss, method="mean")
        stats = {f"{tracker_folder}/{loss_type}-loss": avg_loss}

        # print and log results on main process
        tracker.log(stats, step=trainer.step.item() + 1)

        return avg_loss
    


# Xây dựng hàm report_cosine_sim để tính toán độ tương đồng cosine 
# giữa các embedding hình ảnh và các dữ liệu văn bản trong quá trình đào tạo 
def report_cosine_sims(
    trainer: DiffusionPriorTrainer,
    dataloader: DataLoader,
    text_conditioned: bool,
    tracker: Tracker,
    split: str,
    timesteps: int,
    tracker_folder: str,
):
    # chuyển trình đào tạo sang chế độ xác thực 
    trainer.eval()
    # kiểm tra xem có đnag ở trên quá trình chính hay không 
    if trainer.accelerator.is_main_process:
        # sử dụng hàm click.secho để in ra một thông báo ròng lệnh
        click.secho(
            f"Measuring Cosine-Similarity on {split} split with {timesteps} timesteps",
            # kiểu màu = greeen 
            fg="green",
            blink=True,
        )

    # lấy ra các test_image_embedding và text_data trong trình tải dữ liệu 
    for test_image_embeddings, text_data in dataloader:
        # sau đó thực hiện chuyển các nhúng hinhf ảnh thử nghiệm cho thiết bị
        test_image_embeddings = test_image_embeddings.to(trainer.device)
        # áp dụng tương tự với các text_data 
        text_data = text_data.to(trainer.device)

        #  kiểm tra xem có điều kiện hóa văn bản không 
        if text_conditioned: 
            # tạo ra các nhúng từ văn bản được mã hóa 
            text_embedding, text_encodings = trainer.embed_text(text_data)
            # sau đó lưu các nhúng và mã hóa văn bản vào từ điển text_cond 
            text_cond = dict(text_embed=text_embedding, text_encodings=text_encodings)
        # Trường hợp không có điều kiện hóa văn bản 
        else:
            # gán cho text_embed = text_dat
            text_embedding = text_data
            # và lưu trữ kết quả này vào từ điển text_cond 
            text_cond = dict(text_embed=text_embedding)

        # Tạo một bản sao của phàn nhúng văn bản được sư dụng để xáo trộn
        text_embed_shuffled = text_embedding.clone()

        # cuộn văn bản để mô tả chú thích không liên quan
        # rolled_idx là một tensor được tạo ra bằng cách “lăn” (hoặc dịch chuyển) các chỉ số của tensor text_embedding một vị trí. 
        # Điều này tạo ra một dãy các chỉ số mới mà khi sử dụng để chỉ mục, sẽ “xáo trộn” các embeddings văn bản.
        rolled_idx = torch.roll(torch.arange(text_embedding.shape[0]), 1)
        # text_embed_shuffled là tensor chứa các embeddings văn bản đã được xáo trộn dựa trên rolled_idx
        text_embed_shuffled = text_embed_shuffled[rolled_idx]
        # Cuối cùng, text_embed_shuffled được chuẩn hóa để mỗi embedding có độ dài là 1, 
        # điều này thường được thực hiện trước khi tính toán độ tương đồng cosine.
        text_embed_shuffled = text_embed_shuffled / text_embed_shuffled.norm(
            dim=1, keepdim=True
        )
        # . Nếu text_conditioned là True, tức là nếu quá trình huấn luyện phụ thuộc vào văn bản, 
        # thì text_encodings_shuffled sẽ được gán giá trị là text_encodings đã được xáo trộn theo rolled_idx.
        if text_conditioned:
            text_encodings_shuffled = text_encodings[rolled_idx]
        # còn lại gán cho text_encodings_shuffled = None 
        else:
            text_encodings_shuffled = None

        # xây dựng một từ điển text_cond_shuffled để luu trữ các kết quả của nhung văn bản và mã hóa văn bản 
        text_cond_shuffled = dict(
            text_embed=text_embed_shuffled, text_encodings=text_encodings_shuffled
        )

        # prepare the text embedding
        text_embed = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        # prepare image embeddings
        test_image_embeddings = test_image_embeddings / test_image_embeddings.norm(
            dim=1, keepdim=True
        )

        # predict on the unshuffled text embeddings
        # dự đoán các nhúng văn bản không xáo trộn 
        predicted_image_embeddings = trainer.p_sample_loop(
            # Lặp với số lần bằng với test_image_embedding.shape 
            test_image_embeddings.shape,
            # sử dụng điều kiện hóa văn bản 
            text_cond,
            timesteps=timesteps,
        )

        # SSau đó thực hiện phép chuẩn hóa cho dữ liệu 
        predicted_image_embeddings = (
            predicted_image_embeddings
            / predicted_image_embeddings.norm(dim=1, keepdim=True)
        )

        # predict on the shuffled embeddings
        # dự đoán trên các nhúng được xáo trộn 
        predicted_unrelated_embeddings = trainer.p_sample_loop(
            test_image_embeddings.shape,
            text_cond_shuffled,
            timesteps=timesteps,
        )

        # Tương tự như trên thực hiện chuẩn hóa các kết quả 
        predicted_unrelated_embeddings = (
            predicted_unrelated_embeddings
            / predicted_unrelated_embeddings.norm(dim=1, keepdim=True)
        )


        # 
        # calculate similarities 
        # Tính toán độ tương tự của sảnh sinh tọa 
        orig_sim = pad_gather_reduce(
            # Tính toán độ tương tự nguyên bản sử dụng hàm pad_gather_reduce để 
            # tính toán trung bình cho text_emb và test_image_embeddings 
            trainer, cos(text_embed, test_image_embeddings), method="mean"
        )
        # Tính toán ướcc lượng tương tự 
        pred_sim = pad_gather_reduce(
            # cho trung bình kết quả của text_embed và ảnh được dự đoán
            trainer, cos(text_embed, predicted_image_embeddings), method="mean"
        )
        # Tính toán độ tương tự của text_embed và dự đoán không nhúng văn bản 
        unrel_sim = pad_gather_reduce(
            trainer, cos(text_embed, predicted_unrelated_embeddings), method="mean"
        )
        # tính toán độ tương tự của hình ảnh thư nghiêm và hình ảnh được dự đoán 
        pred_img_sim = pad_gather_reduce(
            trainer,
            cos(test_image_embeddings, predicted_image_embeddings),
            method="mean",
        )

        # SỬ DỤNG TỪ ĐIỂN ĐỂ LƯU TRỮ CÁC TỆP THÔNG TIN VỀ CÁC GIÁ TRỊ SIMILARITY ĐÃ ĐƯỢC TÍNH TOÁN Ở TRÊN 
        stats = {
            f"{tracker_folder}/baseline similarity [steps={timesteps}]": orig_sim,
            f"{tracker_folder}/similarity with text [steps={timesteps}]": pred_sim,
            f"{tracker_folder}/similarity with original image [steps={timesteps}]": pred_img_sim,
            f"{tracker_folder}/similarity with unrelated caption [steps={timesteps}]": unrel_sim,
            f"{tracker_folder}/difference from baseline similarity [steps={timesteps}]": pred_sim
            - orig_sim,
        }

        # SỬ DỤNG TRÌNH THEO DÕI TRACKER ĐỂ GHI NHẬT KÝ THÔNG TIN TỪ ĐIỂN STATS 
        tracker.log(stats, step=trainer.step.item() + 1)


# xÂY DỰNG HÀM VALUATION ĐỂ ĐÁNH GIÁ MÔ HÌNH 
# HÀM NÀY SẼ CHẠY XÁC THỰC TRÊN MỘT MÔ HÌNH VÀ CÁC SỐ LIỆU THỦ THUẬT 
# VÀ TRẢ VỀ LOSS NẾU NÓ ĐƯỢC YÊU CẦU 
def eval_model(
    trainer: DiffusionPriorTrainer,
    dataloader: DataLoader,
    text_conditioned: bool,
    split: str,
    tracker: Tracker,
    use_ema: bool,
    report_cosine: bool,
    report_loss: bool,
    timesteps: List[int],
    loss_type: str = None,
):
    """
    Run evaluation on a model and track metrics

    returns: loss if requested
    """
    # CHUYỂN MÔ HÌNH SANG CHẾ ĐỘ EVAL
    trainer.eval()

    # GÁN USE_EMA = 'EMA' NẾU NHƯ NÓ ONLINE
    use_ema = "ema" if use_ema else "online"
    # KHỞI TẠO MỘI CHUỖI ĐƯỜNG DẪN CHO TRÌNH THEO DÕI TRACKER 
    tracker_folder = f"metrics/{use_ema}-{split}"

    # detemine if valid timesteps are passed
    # TíNH TOÁN SỐ BƯỚC THỜI GIAN TỐI THIỂU CHO VIỆC LẤY MÂŨ 
    min_timesteps = trainer.accelerator.unwrap_model(
        trainer.diffusion_prior
    ).sample_timesteps
    # TƯƠNG TỰ NHƯ TRÊN ĐỂ TÍNH TOÁN CHO SỐ LƯỢNG THỜI GIAN TỐI ĐAĐA
    max_timesteps = trainer.accelerator.unwrap_model(
        trainer.diffusion_prior
        # HÀM NOISE_SCHEDULE.NUM_TIMESTEPS ĐƯỢC SỬ DỤNG ĐỂ TÍNH TOÁN SỐ BƯỚC
        # THỜI GIAN CHO VIỆC THỰC HIỆN NHIỄU 
    ).noise_scheduler.num_timesteps

    # ĐẢM BẢO RẰNG CÁC BƯỚC THỜI GIAN NẰM Ở GIỮA KHOẢNG GIỚI HẠN LOWER VÀ UPPER 
    assert all_between(
        timesteps, lower_bound=min_timesteps, upper_bound=max_timesteps
    ), f"all timesteps values must be between {min_timesteps} and {max_timesteps}: got {timesteps}"

    # measure cosine metrics across various eta and timesteps
    # NẾU NHƯ THAM SỐ REPORT_COSINE CÓ TỒN TẠI
    if report_cosine:
        # DUYỆT QUA SỐ BƯỚC THỜI GIAN 
        for timestep in timesteps:
            # tÍNH TOÁN ĐỘ TƯƠNG TỰ COSINE QUA MỖI BƯỚC THỜI GIAN 
            report_cosine_sims(
                trainer,
                dataloader=dataloader,
                text_conditioned=text_conditioned,
                tracker=tracker,
                split=split,
                timesteps=timestep,
                tracker_folder=tracker_folder,
            )

    # measure loss on a seperate split of data
    # NẾU NHƯ CÓ TỒN TẠI THAM SÔ REPORT_LOSS THAM SỐ NÀY CHO BIẾT CÓ TRẢ VỀ LOSS 
    # XÁC THỰC HAY KHÔNG 
    if report_loss:
        # SỬ DỤNG HÀM REPORT_VALIDATION_LOSS ĐỂ TÍNH TOÁN GIÁ TRỊ LOSS NÀY 
        loss = report_validation_loss(
            trainer=trainer,
            dataloader=dataloader,
            text_conditioned=text_conditioned,
            use_ema=use_ema,
            tracker=tracker,
            split=split,
            tracker_folder=tracker_folder,
            loss_type=loss_type,
        )
        # CUỐI CÙNG TRẢ VỀ LOSS NÀY 
        return loss
    

# Xây dựng phương thức Train để thực hiện đào tạo mô hình 
def train(
    trainer: DiffusionPriorTrainer,
    tracker: Tracker,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    test_loader: DataLoader,
    config: DiffusionPriorTrainConfig,
):
    # init timers
    # khởi tạo các cấu hình thời gian 
    save_timer = Timer()  # when to save
    samples_timer = Timer()  # samples/sec
    validation_profiler = Timer()  # how long is validation taking
    validation_countdown = Timer()  # when to perform evalutation

    # keep track of best validation loss
    # Đặt trình theo dõi của chi phí xác thực tốt nhất 
    best_validation_loss = config.train.best_validation_loss
    samples_seen = config.train.num_samples_seen

    # do training
    # Thực hiện công việc huấn luyện 

    start_epoch = config.train.current_epoch

    # Duyệt qua dnah sách các kỹ nguyên huấn lyện từ kỷ nguyên bắt đầu 
    # đến kỷ nguyên cuối cùng 
    for epoch in range(start_epoch, config.train.epochs):
        # Nếu như chúng ta hoàn thanh một kỹ nguyên cũ và đặt lại phân phối 
        # thành một kỹ nguyên đầy đủ 
        tracker.log({"tracking/epoch": epoch}, step=trainer.step.item())

        # Nếu kỷ nguyên huấn luyện bắt đầu hiện tại = start_epoch + 1 
        # cùng với bước tải dữ liệu > 0
        if train_loader.dataset.get_start() > 0 and epoch == start_epoch+1:
            # Nếu như đang sử dụng tiến trình chính
            if trainer.accelerator.is_main_process:
                # sử dụng hàm click.secho để ghi một thông báo dòng lệnh 
                click.secho(f"Finished resumed epoch...resetting dataloader.")
            # Đặt lại giá trị bắt đầu của tập dữ liệu về 0 thông qua phương thức 
            # dataset.set_start()
            train_loader.dataset.set_start(0)


        # duyệt qua dnah sách các hình ảnh và tiêu đề văn bản trong trình tải dữ liệu 
        for img, txt in train_loader: 
            # Thiết lập mọi thứ từng bước 

            # Thiết lập trình huấn luyện 
            trainer.train()
            # Đặt bước thời gian hiện tại 
            current_step = trainer.step.item()
            # Đăth lại bước thời gian hiện tại 
            samples_timer.reset()

            # Đặt giữ liệu trên thiết bị 
            img = img.to(trainer.device)
            txt = txt.to(trainer.device)


            # Chuyển văn bản và hình ảnh cho model và thực hiện tính toán loss 
            loss = trainer(text=txt, image_embed=img)

            # Biểu diễn việc lan chuyển ngược và áp dụng cập nhật chi phí tổng hợp EMA 
            trainer.update()

            # Tập hợp lại tất cả các thông tin về tất cả các bước huấn luyện 
            # 1: Tập trung tất cả các loss 
            all_loss = pad_gather_reduce(trainer, loss, method="mean")
            # 2: Tính toán số lượng mẫu 
            num_samples = pad_gather_reduce(trainer, len(txt), method="sum")
            # Tính toán số lượng mẫu mỗi giây 
            samples_per_sec = num_samples / samples_timer.elapsed()
            # Số lượng mẫu được tạo ra 
            samples_seen += num_samples
            # và tỷ lệ phân dã tham số cho chi phí tổng hợp 
            ema_decay = trainer.ema_diffusion_prior.get_current_decay()


            # Log 
            # SỬ dụng trình theo dõi tracker để ghi nhật ký log
            # sử dụng một từ điển để lưu trữ các thông tin về samples_per_sec 
            # samples_seen, ema_decay, all_loss
            tracker.log(
                {
                    "tracking/samples-sec": samples_per_sec,
                    "tracking/samples-seen": samples_seen,
                    "tracking/ema-decay": ema_decay,
                    f"tracking/training-{config.prior.loss_type}": all_loss,
                }, 
                # và đặt step = current_step 
            )

            # Theo giõi số liệu và khoảng thời gian được tính 
            eval_delta = pad_gather_reduce(
                # phương thức calidation_countdown.elapsed() sẽ trả vê thời gian đã trôi qua 
                # từ một điểm bắt đầu nào đó, method ='min' cho phép tìm giá trị nhỏ nhất trong 
                # tất cả các giá trị thu thập được 
                trainer, validation_countdown.elapsed(), method="min"
            )
            
            # kiểm tra xem tham số eval_delta có tồn tại và tham số này có lớp 
            # hơn tham sối eval_every_seconds từ cấu hình dữ liệu hay không 
            if eval_delta != None and eval_delta > config.data.eval_every_seconds: 
                # Nếu như điều kiện này đúng 
                # Bắt đầu tính toán thời gian việc này mất bao lâu 

                # Đặt lại thời điểm bắt đầu của chế độ validation 
                validation_profiler.reset()

                # Cấu hình một từ điển kwargs để thực hiện đánh giá 
                eval_kwargs = {
                    "trainer": trainer, 
                    "tracker": tracker, 
                    "text_conditioned": config.prior.condition_on_text_encodings, 
                    "timesteps": config.train.eval_timesteps, 
                }

                # ONLINE MODEL : COSINE : LOSS : VALIDATION SPLIT
                # Cấu hình model Eval với Các cấu hình được truyền vào 
                eval_model(
                    # validation dataloader 
                    dataloader=eval_loader,
                    # config loss 
                    loss_type=config.prior.loss_type,
                    # validation 
                    split="validation",
                    # ema 
                    use_ema=False,
                    # cosine similarity 
                    report_cosine=False,
                    # Use loss 
                    report_loss=True,
                    **eval_kwargs,
                )

                # EMA MODEL : COSINE : LOSS : VALIDATION DATA
                # Xây dựng eval_model 
                ema_val_loss = eval_model(
                    dataloader=eval_loader,
                    loss_type=config.prior.loss_type,
                    split="validation",
                    use_ema=True,
                    report_cosine=True,
                    report_loss=True,
                    **eval_kwargs,
                )

                # Sử dụng trình theo dõi tracker để ghi lại nhật ký 
                # sau mỗi 60 phút 
                tracker.log(
                    {
                        "tracking/validation length (minutes)": validation_profiler.elapsed()
                        / 60
                    }
                )

                # Kiểm tra xem xác thực ema có phải là mức thấp nhất chưa 

                if ema_val_loss < best_validation_loss: 
                    # Nếu điều kiện đúng gán mức thấp nhất = ema_loss 
                    best_validation_loss = ema_val_loss

                    # Bắt đầu lưu trữ mô hình tốt nhất 
                    save_trainer(
                        trainer = trainer , 
                        tracker = tracker, 
                        is_best= True , 
                        is_latest= False , 
                        # số mẫu được quan sát 
                        samples_seen=samples_seen,
                        epoch=epoch,
                        # Và chi phí tốt nhất 
                        best_validation_loss=best_validation_loss,
                    )
                
                # Đặt lại thời gian cho xác thực 

                validation_countdown.reset()

            # KiỂM TRA XEM EVAL_DATA THAM SỐ NÀY CÓ TỒN TẠI 
            elif eval_delta is None:
                    # sử dụng hàm click.secho để in ra một thông báo dòng lệnh 
                    click.secho(
                        f"Error occured reading the eval time on rank: {trainer.device}",
                        fg="yellow",
                    )

            # save as latest model on schedule
            # Lưu trữ dưới dạng mô hình mới nhất theo lịch trình 
            save_delta = pad_gather_reduce(trainer, save_timer.elapsed(), method="min")

            # Kiểm tra xem save_delta != None và lớp hơn tham số lưu trữ mỗi giây 
            if save_delta != None and save_delta >= config.train.save_every_seconds:
                # Nếu như điều kiện này đúng thực hiện lưu trữ mô hình 
                # huấn luyện 
                save_trainer(
                    trainer=trainer,
                    tracker=tracker,
                    is_best=False,
                    is_latest=True,
                    samples_seen=samples_seen,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                )
                # Đánh giấu thời gian 
                save_timer.reset()

            # Trường hợp còn lại tức là save_delta không tồn tại 
            elif save_delta is None:
                # In ra một thông báo dòng lệnh 
                click.secho(
                    f"Error occured reading the save time on rank: {trainer.device}",
                    fg="yellow",
                )

    # evaluate on test data
    # Nếu như đang ở trên quá trình chính 
    if trainer.accelerator.is_main_process:
        # In ra một thông báo dòng lệnh 
        click.secho(f"Starting Test", fg="red")

    # save one last time as latest before beginning validation
    # Lưu lần cuối là mới nhất trươcs khi bắt đầu xác thực 
    save_trainer(
        tracker=tracker,
        trainer=trainer,
        is_best=False,
        is_latest=True,
        samples_seen=samples_seen,
        epoch=epoch,
        best_validation_loss=best_validation_loss,
    )
    # Bắt đầu mô hình xác thực 
    test_loss = eval_model(
        trainer=trainer,
        dataloader=test_loader,
        text_conditioned=config.prior.condition_on_text_encodings,
        split="test",
        tracker=tracker,
        use_ema=True,
        report_cosine=False,
        report_loss=True,
        timesteps=config.train.eval_timesteps,
        loss_type=config.prior.loss_type,
    )
    # Kiểm tra xem loss_test có nhỏ hơn chi phí thấp nhất
    if test_loss < best_validation_loss:
        # nẾU CÓ GÁN LẠI CHI PHÍ TỐI ƯU = TEST_LOSS
        best_validation_loss = test_loss

        #  go save the model as best
        # Thực hiện lưu trữ lại mô hình tốt nhất 
        save_trainer(
            trainer=trainer,
            tracker=tracker,
            is_best=True,
            is_latest=False,
            samples_seen=samples_seen,
            epoch=epoch,
            best_validation_loss=test_loss,
        )

            
def initialize_training(config_file, accelerator):
    """
    Parse the configuration file, and prepare everything necessary for training
    """
    # load the configuration file
    if accelerator.is_main_process:
        click.secho(f"Loading configuration from {config_file}", fg="green")

    config = TrainDiffusionPriorConfig.from_json_path(config_file)

    # seed

    set_seed(config.train.random_seed)

    # get a device

    device = accelerator.device

    # make the trainer (will automatically distribute if possible & configured)

    trainer: DiffusionPriorTrainer = make_model(
        config.prior, config.train, device, accelerator
    ).to(device)

    # create a tracker

    tracker = create_tracker(
        accelerator, config, config_file, dummy=accelerator.process_index != 0
    )

    # reload from chcekpoint

    if tracker.can_recall:
        current_epoch, best_validation_loss, samples_seen = recall_trainer(
            tracker=tracker, trainer=trainer
        )

        # display best values
        if trainer.accelerator.is_main_process:
            click.secho(f"Current Epoch: {current_epoch} | Best Val Loss: {best_validation_loss} | Samples Seen: {samples_seen}", fg="yellow")

        # update config to reflect recalled values
        config.train.num_samples_seen = samples_seen
        config.train.current_epoch = current_epoch
        config.train.best_validation_loss = best_validation_loss

    # fetch and prepare data

    if trainer.accelerator.is_main_process:
        click.secho("Grabbing data...", fg="blue", blink=True)

    trainer.accelerator.wait_for_everyone()
    img_reader = get_reader(
        text_conditioned=trainer.text_conditioned,
        img_url=config.data.image_url,
        meta_url=config.data.meta_url,
    )

    # calculate start point within epoch

    trainer.accelerator.wait_for_everyone()

    train_loader, eval_loader, test_loader = make_splits(
        text_conditioned=trainer.text_conditioned,
        batch_size=config.data.batch_size,
        num_data_points=config.data.num_data_points,
        train_split=config.data.splits.train,
        eval_split=config.data.splits.val,
        image_reader=img_reader,
        rank=accelerator.state.process_index,
        world_size=accelerator.state.num_processes,
        start=0,
    )

    # update the start point to finish out the epoch on a resumed run

    if tracker.can_recall:
        samples_seen = config.train.num_samples_seen
        length = (
            config.data.num_data_points
            if samples_seen <= img_reader.count
            else img_reader.count
        )
        scaled_samples = length * config.train.current_epoch
        start_point = (
            scaled_samples - samples_seen if scaled_samples > samples_seen else samples_seen
        )

        if trainer.accelerator.is_main_process:
            click.secho(f"Resuming at sample: {start_point}", fg="yellow")

        train_loader.dataset.set_start(start_point)

    # start training

    if trainer.accelerator.is_main_process:
        click.secho(
            f"Beginning Prior Training : Distributed={accelerator.state.distributed_type != accelerate_dataclasses.DistributedType.NO}",
            fg="yellow",
        )

    train(
        trainer=trainer,
        tracker=tracker,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader,
        config=config,
    )

@click.command()
@click.option("--config_file", default="configs/train_prior_config.example.json")
def main(config_file):
    # start HFA
    accelerator = Accelerator()

    # setup training
    initialize_training(config_file, accelerator)


if __name__ == "__main__":
    main()