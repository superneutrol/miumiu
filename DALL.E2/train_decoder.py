from pathlib import Path 
from typing import List 
from datetime import timedelta 

from dalle2_pytorch.trainer import DecoderTrainer
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader 
from dalle2_pytorch.trackers import Tracker 
from dalle2_pytorch.train_configs import DecoderConfig, TrainDecoderConfig 
from dalle2_pytorch.utils import Timer, print_ribbon 
from dalle2_pytorch.dalle2_pytorch import Decoder, resize_image_to 
from clip import tokenize 

import torchvision 
import torch 
from torch import nn 
from torchmetrics.image.fid import FrechetInceptionDistance 
from torchmetrics.image.inception import InceptionScore 
from torchmetrics.image.kid import KernelInceptionDistance 
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import dataclasses as accelerate_dataclasses 

import webdataset as wds 
import click 


# Define constants 
TRAIN_CALC_LOSS_EVERY_ITERS = 10
VALID_CALC_LOSS_EVERY_ITERS = 10 

# HELPER FUNTION 
# Hàm exists trả về chính giá trị đầu vào nếu như 
# nó có tồn tại 
def exists(val):
    return val is not None 


# Xây dựng hàm Dataloader Hàm này được sử dụng để tải dữ liệu 
# cho việc đào tạo mô hình decoder 
def create_dataloaders(
    available_shards, webdataset_base_url,
    img_embeddings_url=None, text_embeddings_url=None,
    shard_width=6, num_workers=4, batch_size=32,
    n_sample_images=6, shuffle_train=True, resample_train=False,
    img_preproc = None, index_width=4, train_prop = 0.75,
    val_prop = 0.15, test_prop = 0.10, seed = 0, **kwargs
    ):

    """Randomly splits the availabel shards into train, val, and test sets and returns a dataloader for 
    each
    """
    # đảm bảo rằng tổng của 3 tỷ lệ cắt dữ liệu cho nhiệm vụ train, test, val == 1 
    assert train_prop + test_prop + val_prop == 1 
    # tính toán số lượng mẫu cho dữ liệu đào tạo hàm round được sử dụng để làm tròn kết quả 
    # theo tỷ lệ train_prop 
    num_train = round(train_prop * len(available_shards))
    # tương tự với số lượng mẫu cho dữ liệu thử nghiệm và làm tròn theo 
    # tích với tỷ lệ test_prop
    num_test = round(test_prop * len(available_shards))
    # cuối cùng ta thực hiện tương tự với sô lượng dữ liệu cho việc xác thực mô hình 
    # được tính toán bằng tổng số lượng mẫu theo danh sách shard - num_test, num_train 
    num_val = len(available_shards) - num_train - num_test 
    # đảm bảo rằng số lượng mẫu theo 3 danh sách này bằng với danh sách shard 
    assert num_train + num_test + num_val == len(available_shards), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(available_shards)}"
    # tách ra các phần dữ liệu được chỉ định theo số lượng đã tính toán theo danh sách shard 
    # hàm generator = torch.Generator().mannual_seed đảm bảo tính nhất quán bằng cách sử dụng một seed 
    # cố định cho việc tạo số ngẫu nhiên 
    train_split, test_split, val_split = torch.utils.data.random_split(available_shards, [num_train, num_test, num_val], generator=torch.Generator().manual_seed(seed))

    # The shard number in the webdataset file names has a fixed width. We zero pad the shard numbers so they correspond to a filename.
    # Xây dựng các danh sách URL cho từng phần tách bằng cách sử dụng webdataset_base_url và số shard 
    train_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in train_split]
    # SỬ dụng hàm zerofill để đệm các phần shard còn thiếu từ danh sách shard thành shard có độ dài tiêu chuẩn 
    test_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in test_split]
    # cuối cùng áp dụng với validation 
    val_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in val_split]

    # Xây dựng DataLoader hàm này được định nghĩa dưới dạng biểu thức Lambda 
    create_dataloader = lambda tar_url, shuffle=False, resample=False, for_sampling=False: create_image_embedding_dataloader(
        # Hàm này tạo một DataLoader cho dữ liệu nhúng hình ảnh sử dụng các tham số đã chỉ định 
        tar_url=tar_url, 
        # num_worker tham số này chỉ định số các tiến trình con được sử dụng để tải dữ liệu
        num_workers=num_workers, 
        # tham số batch_size được gán = batch_size nếu tham số for_sampling không tồn tại 
        # còn lại gán = n_smaple_images chỉ định số lượng mẫu cho mỗi lần
        batch_size=batch_size if not for_sampling else n_sample_images,
        # url nhúng hình ảnh 
        img_embeddings_url=img_embeddings_url, 
        # Url nhúng văn bản
        text_embeddings_url=text_embeddings_url,
        # chỉ số idx theo chiều hàng 
        index_width=index_width,
        # shuffle num chỉ định số lượng mẫu sẽ được xáo chộn 
        shuffle_num = None,
        # keys trích xuất là txt
        extra_keys= ["txt"],
        # tham số xáo chộn cho shard 
        shuffle_shards = shuffle,
        # Resample_shard tham số lấy mẫu lại cho shard hay là tái tạo lại shard 
        resample_shards = resample, 
        # và tham số image_prepprocess đây là hàm tiền xử lý hình ảnh 
        img_preproc=img_preproc,
        # và handle Xử lý thông báo cảnh báo 
        handler=wds.handlers.warn_and_continue
    )

    # Tạo DataLoader cho tập huấn luyện và đặt cho nó có thể xáo chộn và tái tạo lại 
    train_dataloader = create_dataloader(train_urls, shuffle=shuffle_train, resample=resample_train)
    # Tạo DataLoader cho Mục Mẫu Huấn Luyện (Train Sampling):
    # Tạo một Dataloader dành cho việc lấy mẫu từ tập huấn luyện
    train_sampling_dataloader = create_dataloader(train_urls, shuffle_train=False, for_sampling=True)
    # Tạo một Validation Dataloader cho tập xác thực 
    val_dataloader = create_dataloader(val_urls, shuffle=False)
    # Tạo một Test Dataloader cho tập thử nghiệm
    test_dataloader = create_dataloader(test_urls, shuffle=False)
    # Tạo một Dataloader cho mục mẫu kiểm tra 
    # điều này cho phép tạo một Dataloader dành riêng cho việc lấy mẫu từ tập kiểm tra 
    test_sampling_dataloader = create_dataloader(test_urls, shuffle=False, for_sampling=True)
    # cuối cùng đoạn mã trả về một từ điển chứa các DataLoader cho các tập dữ liệu khác nhau 
    return {
        "train": train_dataloader, 
        "train_sampling": train_sampling_dataloader, 
        "val": val_dataloader, 
        "test": test_dataloader, 
        "test_sampling": test_sampling_dataloader
    }


# Xây dựng hàm get_data_keys hàm này sẽ lấy ra keys
# từ trình tải dữ liệu để có thể thực hiện việc trích xuất dữ liệu từ bộ 
# tải dữ liệu thực 
def get_dataset_keys(dataloader):
    """
    It is sometimes neccesary to get the keys the dataloader is returning. Since the dataset is burried in the dataloader, we need to do a process to recover it.
    """
    # If the dataloader is actually a WebLoader, we need to extract the real dataloader
    # đảm bảo rằng dataloader trình tải dữ liệu này là một WebLoader 
    if isinstance(dataloader, wds.WebLoader):
        # sử dụng đường ống dữ liệu lấy ra ph
        dataloader = dataloader.pipeline[0]
        # return dataloader.dataset.key_map: Cuối cùng, hàm trả về key_map từ tập dữ liệu (dataset) của DataLoader. 
        # key_map là một thuộc tính chứa thông tin về cách các khóa được ánh xạ đến dữ liệu trong tập dữ liệu.
    return dataloader.dataset.key_map


# Xây dựng hàm get_xample_data để thực hiện lấy mẫu 
# dữ liệu và trả về một danh sách các mẫu được nén 
def get_example_data(dataloader, device, n=5):
    """
    Samples the dataloader and returns a zipped list of examples
    """
    # Xây dựng 3 danh sách để lưu trữ 
    # 1: Hình ảnh thô 
    images = []
    # 2: Danh sách chứa các nhúng hình ảnh 
    img_embeddings = []
    # 3: Và dnah sách chứa các chúng văn bản 
    text_embeddings = []
    # Và một Danh sách để nhúng các tiêu đề văn bản
    captions = []

    # duyệt qua qua danh sách trình tải lấy ra các embedding của văn bản và hình ảnh 
    # đồng thời lấy ra hình ảnh và văn bản 
    for img, emb, txt in dataloader: 
        # lấy ra các nhúng hình ảnh và văn bản 
        img_emb, text_emb = emb.get('img'), emb.get('text')
        # kiểm tra xem danh sách img_emb = None 
        if img_emb is not None: 
            # Nếu danh sách này # None 
            # chuyển danh sách này cho GPU 
            img_emb = img_emb.to(device=device, dtype=torch.float)
            # sử dụng hàm list để chuyển đổi các img_emb thành accs danh sách sau đó 
            # ta sử dụng hàm extend để nối các danh sách này với nhau 
            img_embeddings.extend(list(img_emb))

        # Nếu như danh sách này rỗng 
        else: 
            # sau đó thêm 1 None img.shape[0] vào danh sách img_embedding 
            img_embeddings.extend([None]*img.shape[0])
        
        # Thực hiện tương tự với text_embed
        if text_emb is not None: 
            # chuyển các text_embedding cho thiết bị 
            text_emb = text_emb.to(device=device, dtype=torch.float)
            # sau đó sử dụng hàm list để chuyển đổi các text_embedidng thành các dnah sách 
            # sau đó sử dụng extend để thêm chúng vào danh sách img_embedding 
            img_embeddings.extend(list(text_emb))

        # Trường hợp còn lại 
        else: 
            # Chúng ta thêm một danh sách None có kích thước bằng img.shape[0]
            text_embeddings.extend([None]*img.shape[0])
        
        # sau đó chuyển các hình ảnh được trích xuất từ trình tải cho GPU 
        img = img.to(device= device, dtype=torch.float)
        # sau đó sử dụng list để chuyển đổi các img thành danh sách và nối chúng bằng 
        # extend để them vào danh sách lưu  trữ images
        images.extend(list(img))
        # với các tiêu đề văn bản ta thêm chúng trực tiếp vào danh sách dưới dạng các danh sách 
        captions.extend(list(txt))
        # kiểm tra xem nếu danh sách image >= 5 hay không 
        if len(images) >= n:
            # nếu thỏa mãn thì dừng lại 
            break 
    
    # trả về một danh sách chứa các danh sách text_embedding , img, captions 
    # và imge_embeding được cắt từ 0 -> n mẫu 
    # hàm zip được sử dụng để nén một loạt các danh sách thành một danh sách trả về duy nhất '
    return list(zip(images[:n], img_embeddings[:n], text_embeddings[:n], captions[:n]))



# Thiết lập phương thức generate_sample phương thức này nhận các mẫu dữ liệu 
# và tạo các hình ảnh từ các biểu diễn nhúng 
# Hàm này trả về 3 danh sách gômg real_image [ảnh thực], generated_image[ảnh được tạo ra] và captions[các tiêu đề văn bản của hình ảnh]
def generate_samples(trainer, example_data, clip=None, start_unet=1, end_unet=None, 
            condition_on_text_encodings=False, cond_scale=1.0, device=None, text_prepend="", match_image_size=True):
    """
    Takes example data and generates images from the embeddings
    Returns three lists: real images, generated images, and captions
    """
    # sử dụng hàm zip đển phân giải một danh sách được nén example_data
    # thành 4 danh sách gồm real_image, image_embed, text_embed, captions
    real_images, img_embeddings, text_embeddings, txts = zip(*example_data)
    # và một từ điển params để lưu trữ các kết quả
    sample_params = {}
    # kiểm tra xem số lượng các phần tử trong danh sách img_embedding có = 0
    if img_embeddings[0] is None: 
        # Tạo các ảnh nhúng từ clip 
        # đầu tiên nối dnah sách hoặc tensor real_images băng hàm stack 
        imgs_tensor = torch.stack(real_images)
        # đảm bảo rằng có tồn tại mô hình CLIP 
        assert clip is not None, "clip is None, but img_embeddings is None"
        # chuyển tensor imgs cho thiết bị GPU có thể là TPU
        imgs_tensor.to(device=device)
        # kêu gọi hàm embed_images để thực hiện nhúng cho hình ảnh thực 
        # kết quả là 1 tuple gồm img_embedding biểu diễn các nhúng hình ảnh 
        # và img_encoding thể hiện hình ảnh được giải mã
        img_embeddings, img_encoding = clip.embed_image(imgs_tensor)
        # Thêm các img_embeding vào từ điển params với keys = image_embed
        sample_params["image_embed"] = img_embeddings
    # Trường hợp còn lại không phải None
    else:
        # Then we are using precomputed image embeddings
        # Ta sử dụng tính năng nhúng hình ảnh được tính toán trước để sinh ra các hình ảnh 
        # 1: Hàm satck để nối các dnah sách nhúng hình ảnh 
        img_embeddings = torch.stack(img_embeddings)
        # 2: Thêm vào từ điển params key = image_embed và values = img_embeddings 
        sample_params["image_embed"] = img_embeddings
    # Nếu như có điều kiện hóa văn bản tham số này chỉ định Nếu 
    # như = True ảnh sẽ được tạo dựa trên mô tả văn bản đầu vào không thì ngựoc lại
    if condition_on_text_encodings:
        # kiểm tra xem dnah sahs text_embeeding = None 
        if text_embeddings[0] is None:
            # Generate text embeddings from text
            # Theo đó thực hiện tạo các hình ảnh từ văn bản nhúng 
            assert clip is not None, "clip is None, but text_embeddings is None"
            # sử dụng hàm tokenizer để nhúng các văn bản , tham số truncate chỉ định 
            # việc các các phần văn bản sau đó chuyển chúng cho thiết bị
            tokenized_texts = tokenize(txts, truncate=True).to(device=device)
            # thực hiện nhúng các mã hóa văn bản đê lấy ra các biểu diễn mã hóa và văn nhúng văn bản
            text_embed, text_encodings = clip.embed_text(tokenized_texts)
            # Thêm các biểu diên nhúng vào từ điển params với key = text_encoding 
            # và values = text_encodings
            sample_params["text_encodings"] = text_encodings
        
        # Trường hợp còn lại 
        else:
            # Then we are using precomputed text embeddings
            # sử dụng tính năng nhúng văn bản được tính toán trước
            text_embeddings = torch.stack(text_embeddings)
            # thêm biểu diễn mã hóa văn bản vào từ điển 
            sample_params["text_encodings"] = text_embeddings
    
    # Thêm vào từ điển params 2 tham số là start và end unet 
    sample_params["start_at_unet_number"] = start_unet
    sample_params["stop_at_unet_number"] = end_unet
    # kiểm trả xem start unet > 1 hay không 
    if start_unet > 1:
        # If we are only training upsamplers
        # Chúng ta chỉ đào thông qua lấy mẫu 
        # thêm vào từ điển params các ảnh thực tế
        sample_params["image"] = torch.stack(real_images)
    # kiểm tra thiết bị
    if device is not None:
        # Thêm thông tin thiết vị vào từ điển 
        sample_params["_device"] = device

    # Lấy mẫu sử dụng hàm trainer.sample để lấy mẫu huấn luyện 
    # toán tử **được sử dụng để nén từ điển params 
    # Tại thời điểm lấy mẫy chuyển sao FP16
    samples = trainer.sample(**sample_params, _cast_deepspeed_precision=False)  # At sampling time we don't want to cast to FP16
    # sau đó các mẫu này được lưu trữ thành danh sách gán kết quả cho generated_images
    generated_images = list(samples)
    # sau đó nối chuỗi text_prepend với một một chỗi văn ban trong dnah sách văn abnr hay tiêu đề văn bản 
    captions = [text_prepend + txt for txt in txts]
    # kiểm tra xem kích thước của hình ảnh có phù hợp hay không 
    if match_image_size:
        # generated_images[0] lấy ảnh đầu tiên trong danh sách và .shape[-1] để lấy ra kích thước của ảnh 
        generated_image_size = generated_images[0].shape[-1]
        # Resize_image và đặt chúng trong biểu diễn 0-> 1 hàm clamp_range sẽ giới hạn điều này 
        real_images = [resize_image_to(image, generated_image_size, clamp_range=(0, 1)) for image in real_images]
    # cuối cùng trả về ảng thực , ảnh được tạo và tiêu đề văn bản 
    return real_images, generated_images, captions


# Thiết lập phương thức generate_grid_smaple để biểu diễn các mẫu 
# được lấy qua hàm generate_samples 
def generate_grid_samples(trainer, examples, clip=None, start_unet=1, end_unet=None, condition_on_text_encodings=False, cond_scale=1.0, device=None, text_prepend=""):
    """
    Generates samples and uses torchvision to put them in a side by side grid for easy viewing
    """
    # trích xuất accs biểu diễn từ generate_samples
    real_images, generated_images, captions = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, device, text_prepend)
    # sử dụng hàm torchvision.utils.make_grid để hiển thị các ảnh thạt và giả dưới dnagj lưới gridview 
    grid_images = [torchvision.utils.make_grid([original_image, generated_image]) for original_image, generated_image in zip(real_images, generated_images)]
    return grid_images, captions


# Thiết lập phương thức evaluate_trainer để thực hiện tính toán 
# số liệu đánh giá cho mô hình decoder 
def evaluate_trainer(trainer, dataloader, device, start_unet, end_unet, clip=None, 
            condition_on_text_encodings=False, cond_scale=1.0, inference_device=None, 
            n_evaluation_samples=1000, FID=None, IS=None, KID=None, LPIPS=None):
    """
    Computes evaluation metrics for the decoder
    """
    # Xây dựng một từ điển metrics để lưu trữ các kế quả được tính toán 
    metrics = {}
    # Chuẩn bị dữ liệu 
    # sử dụng hàm get_example để lấy mẫu dữ liệu 
    examples = get_example_data(dataloader, device, n_evaluation_samples)
    # kiểm tra xem các phần tử được lấy ra bởi hàm get_example_data 
    # có > 0 
    if len(examples) == 0:
        # Nếu như danh sách này = 0
        # In ra một thông báo cho biết không tồn tại dữ liệu đánh giá
        print("No data to evaluate. Check that your dataloader has shards.")
        # và lập tức trả về từ điển metrics 
        return metrics
    
    # Thực hiện lấy mẫu các kết quả thông qua hàm generate với các tham số được chuyển vào 
    # bao gồm real_images, generated_images, captions 
    real_images, generated_images, captions = generate_samples(trainer, examples, clip, start_unet, 
                                    end_unet, condition_on_text_encodings, cond_scale, inference_device)
    # sau đó xếp chồng danh sách real_images bằng hàm stack và chuyển việc tính toan cho device 
    real_images = torch.stack(real_images).to(device=device, dtype=torch.float)
    # Thực hiện tương tự với danh sach Generated_images 
    generated_images = torch.stack(generated_images).to(device=device, dtype=torch.float)

    # Chuyển đổi các biểu diễn của hình ảnh [0->] sang định dnagj RGB và từ torch.float -> torch.uint8
    # Đầu tiên với danh sách real_images nhân các phần tử với 255 sau đó + 1/2 và đặt giới 
    # hạn các giá trị 0-> 255 cuối cùng là chuyển no sang biểu diễn torch.uint8
    int_real_images = real_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    # Tương tự với dnah sách generated_images 
    int_generated_images = generated_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)


    # Thiết lập phương thức null_sync pt này sẽ trả về 
    # tensor đầu vào dưới dnag danh sách 
    def null_sync(t, *args, **kwargs):
        # trả về danh sách t 
        return [t]
    
    # Nếu như tồn tại FID (Frechet Inception Distance)
    # đây là một phép đo khoảng cách giữa các phân phối ảnh thực và ảnh được tạo ra 
    # bởi mô hình. nếu FID càng thấp thì chất lượng ảnh càng cao 
    if (FID):
        # Nếu tồn tại giá trị này tạo một đối tượng FID với 
        # các tham số được truyền vào, và thiết lập chức năng đồng bộ hóa khonagr cách là null_sync 
        fid = FrechetInceptionDistance(**FID, dist_sync_fn=null_sync)
        # sau đó chuyển đến thiết bị để có thể thực hiện tính toan 
        fid.to(device)
        # Cập nhật đối tượng FID với ảnh thật và đặt real = True 
        fid.update(int_real_images, real=True)
        # Tương tự cập nhật FID với ảnh fake và đặt real = False 
        fid.update(int_generated_images, real=False)
        # sau đó tính toán khoảng cách sai lệch FID và thêm kết quả và từ điển metrics 
        # vơí key= FID 
        metrics["FID"] = fid.compute().item()
    
    # Nếu như Is có tồn tại (Inception Score) là một phép đp để đánh giá chất lượng 
    # của hình ảnh được tạo ra bởi mô hình . Nó dựa trên việc so sánh phân phối xác 
    # suất của các lớp đối tượng trong ảnh Is càng cao chất lượng càng tốt 
    if (IS):
        # Khởi tạo một đối tương Inception Score với các tham số được truyền vào 
        # và thiết lập chức năng đồng bộ hóa khoảng cách là null_sync 
        inception = InceptionScore(**IS, dist_sync_fn=null_sync)
        # sau đó chuyển đến thiết bị để thực hiện tính toán 
        inception.to(device=device)
        # Cập nhật đối tượng IS với ảnh thật 
        inception.update(int_real_images)
        # sau đó thực hiện tính toán trên các ảnh thật này để có thể có được 
        # std và mean cho mỗi hình ảnh 
        is_mean, is_std = inception.compute()
        # Thêm các phần tử theop độ lệch chuẩn và phươg sai vào từ điển metrics 
        # với keys = is_mean và is_std với các giá trị tương ứng với nó 
        metrics["IS_mean"] = is_mean.item()
        metrics["IS_std"] = is_std.item()
    
    # kiểm tra xem giá trị KID (kernel Inception Distance) có tồn tại hay không 
    # đây là một thước đó khoảng cách để đánh giá mực bộ phân bố của hình ảnh được tạo ra 
    # và ảnh thực 
    if exists(KID):
        # Khởi tạo môt đối tượng Kernel Inception Distance với các tham số được ytruyeenf vào 
        # và thiết lập chức năng đồng bộ hóa khoảng cách 
        kernel_inception = KernelInceptionDistance(**KID, dist_sync_fn=null_sync)
        # chuyển đối tượng KernelInception sang thiết bị tính toán 
        kernel_inception.to(device=device)
        # Cập nhật đối tượng KID với ảnh thực và đặt real = True 
        kernel_inception.update(int_real_images, real=True)
        # Cập nhật đối tượng Kernel Inception Distance với hình ảnh được tạo ra (int_generated_images) và đặt real=False.
        kernel_inception.update(int_generated_images, real=False)
        # Tính toán chỉ số KID (kid_mean và kid_std) 
        kid_mean, kid_std = kernel_inception.compute()
        # và lưu trữ vào từ điển metrics với khóa "KID_mean" và "KID_std".
        metrics["KID_mean"] = kid_mean.item()
        metrics["KID_std"] = kid_std.item()

    # LPIPS cũng là môt thước đo khoảng cách sử dụng để đánh giá 
    # mức độ tương đồng về nhận thức của ảnh thực so với ảnh được tạo ra 
    if exists(LPIPS):
        # Convert from [0, 1] to [-1, 1]
        # Thực hiện một chuyển đổi ảnh thật (real_images) và ảnh được tạo ra generated_images từ 
        # khoảng [0, 1] sang khoảng [-1, 1] bằng cách nhân lên 2 và trừ 1, sau đó giới hạn trong khoảng [-1, 1].
        renorm_real_images = real_images.mul(2).sub(1).clamp(-1,1)
        renorm_generated_images = generated_images.mul(2).sub(1).clamp(-1,1)
        # Tạo một đối tượng Learned Perceptual Image Patch Similarity với các tham số được truyền vào 
        # và thiết lập chức năng đồng bộ hóa khoảng cách là null_sync.
        lpips = LearnedPerceptualImagePatchSimilarity(**LPIPS, dist_sync_fn=null_sync)
        # Chuyển đối tượng Learned Perceptual Image Patch Similarity sang thiết bị tính toán (device).
        lpips.to(device=device)
        # Cập nhật đối tượng Learned Perceptual Image Patch S với ảnh thật và ảnh được tạo 
        # đã được chuyển đổi trước đó 
        lpips.update(renorm_real_images, renorm_generated_images)
        # tính toán chỉ số tương đồng nhận thức và cập nhật tham số này cho từ điển 
        metrics["LPIPS"] = lpips.compute().item()


    # Nếu số lượng quá trình (num_process) của trình tăng tốc (Accelerator) > 
    if trainer.accelerator.num_processes > 1:
        # Then we should sync the metrics 
        # Thì chúng ta nên đồng bộ hóa các chỉ số metrics
        # : Sắp xếp các kháo keys trong từ điển theo thứ tự từ 
        metrics_order = sorted(metrics.keys())
        # Tạo một tensor zeros mới có kích thước là [1, len(metrics)] và kiểu float trên thiết bị 
        # tính toán
        metrics_tensor = torch.zeros(1, len(metrics), device=device, dtype=torch.float)
        # Duyệt qua chỉ số và giá trị của danh sách các khóa 
        for i, metric_name in enumerate(metrics_order):
            # sau đó lấy các khóa trong tensor metrics theo metrics_name 
            # và cập nhật tensor metrics_tensor theo các khóa metrics tại mỗi chỉ số i 
            # tại vị trí hàng 0 cột i 
            metrics_tensor[0, i] = metrics[metric_name]
        
        # sử dụng hàm tập trung gather để tâoj hợp tất cả các tensor metric từ các 
        # quá trình huấn luyện 
        metrics_tensor = trainer.accelerator.gather(metrics_tensor)
        # sau đó tính toán giá trị trinh bìng của tensor metrics theo chiều 0 
        metrics_tensor = metrics_tensor.mean(dim=0)
        # Duyệt qua các khóa của từ điển metrics và lưu trữ giá trị trung bình của mỗi chỉ số vào từ điển metrics.
        for i, metric_name in enumerate(metrics_order):
            # và lưu trữ giá trị trung bình của mỗi chỉ số vào từ điển metrics.
            metrics[metric_name] = metrics_tensor[i].item()
    # và trả về từ điên  metric đẫ được đônhf bộ hóa 
    return metrics

# Xây dựng phương thức save_trainer phương trích này gọi đến trình theo dõi 
# Tracker để ghi nhật ký logs cho mô hình 
def save_trainer(tracker: Tracker, trainer: DecoderTrainer, epoch: int, sample: int, next_task: str, 
                validation_losses: List[float], samples_seen: int, is_latest=True, is_best=False):
    """
    Logs the model with an appropriate method depending on the tracker
    """
    # sử dụng tracker.save để lưu trữ các kết quả của trình đào tạo và ghi 
    # nhật ký các thông tin của mô hình và log
    tracker.save(trainer, is_best=is_best, is_latest=is_latest, epoch=epoch, sample=sample, 
                next_task=next_task, validation_losses=validation_losses, samples_seen=samples_seen)


# Xây dưng phương thức recall phươnh thức này sử dụng để tải mô hình
# Thông qua trình theo dõi tracker 
def recall_trainer(tracker: Tracker, trainer: DecoderTrainer):
    """
    Loads the model with an appropriate method depending on the tracker
    """
    # Sử dụng trình tăng tốc huấn luyện để in ra thông báo 
    # mô hình đang được tải từ .. 
    trainer.accelerator.print(print_ribbon(f"Loading model from {type(tracker.loader).__name__}"))
    # Sau đó, hàm gọi phương pháp recall của đối tượng tracker để 
    # lấy về một từ điển (state_dict) chứa thông tin về mô hình.
    state_dict = tracker.recall()
    # Tiếp theo, hàm gọi phương pháp load_state_dict của đối tượng trainer để tải mô hình từ từ điển state_dict.
    # Tham số only_model=False cho phép tải cả mô hình và các thông số khác, còn tham số strict=True yêu cầu phải tải tất cả các thông số.
    trainer.load_state_dict(state_dict, only_model=False, strict=True)
    # Tham số only_model=False cho phép tải cả mô hình và các thông số khác, còn tham số strict=True yêu cầu phải tải tất cả các thông số.
    return state_dict.get("epoch", 0), state_dict.get("validation_losses", []), state_dict.get("next_task", "train"), state_dict.get("sample", 0), state_dict.get("samples_seen", 0)


# Xây dựng hàm Train được sử dụng để xử lý các tiến trình đào tạo mô 
# hình decoder 
def train(
    # dataloaders: Dataloader cho tập dữ liệu huấn luyện và kiểm tra.
    dataloaders,
    # decoder: Mô hình giải mã.
    decoder: Decoder,
    # accelerator: Đối tượng để tăng tốc huấn luyện (ví dụ: sử dụng GPU).
    accelerator: Accelerator,
    # tracker: Đối tượng để theo dõi các thông số huấn luyện (ví dụ: loss, accuracy
    tracker: Tracker,
    # inference_device: Thiết bị để thực hiện dự đoán (ví dụ: CPU hoặc GPU).
    inference_device,
    clip=None,
    # evaluate_config: Cấu hình để đánh giá mô hình (nếu có).
    evaluate_config=None,
    # epoch_samples: Số lượng mẫu trong mỗi epoch (nếu tập dữ liệu huấn luyện đang tái chọn mẫu).
    epoch_samples = None,  # If the training dataset is resampling, we have to manually stop an epoch
    validation_samples = None,
    # save_immediately: Lưu mô hình ngay sau khi huấn luyện (nếu có).
    save_immediately=False,
    epochs = 20,
    n_sample_images = 5,
    # save_every_n_samples: Lưu mô hình sau mỗi số lượng mẫu nhất định.
    save_every_n_samples = 100000,
    # unet_training_mask: Mặt nạ huấn luyện cho mô hình UNet (nếu có).
    unet_training_mask=None,
    # condition_on_text_encodings: Có điều kiện trên mã hóa văn bản không (nếu có).
    condition_on_text_encodings=False,
    cond_scale=1.0,
    **kwargs
):
    """
    Trains a decoder on a dataset.
    """
    # is_master: Biến kiểm tra xem tiến trình hiện tại có phải là tiến trình chính hay không (ví dụ: tiến trình đầu tiên trong multi-GPU).
    is_master = accelerator.process_index == 0

      # Nếu không tồn tại unet_training_mask,
    if not exists(unet_training_mask):
        # Then the unet mask should be true for all unets in the decoder
        #thì mặc định unet_training_mask sẽ là True cho tất cả các unet trong decoder
        unet_training_mask = [True] * len(decoder.unets)
    
    # # Kiểm tra độ dài của unet_training_mask
    assert len(unet_training_mask) == len(decoder.unets), f"The unet training mask should be the same length as the number of unets in the decoder. Got {len(unet_training_mask)} and {trainer.num_unets}"
    # Xác định các số unet có thể huấn luyện
    trainable_unet_numbers = [i+1 for i, trainable in enumerate(unet_training_mask) if trainable]
    # first_trainable_unet: Đây là số thứ tự của unet đầu tiên có thể huấn luyện.
    # Nó được lấy từ danh sách trainable_unet_numbers.
    first_trainable_unet = trainable_unet_numbers[0]
    # last_trainable_unet: Đây là số thứ tự của unet cuối cùng có thể huấn luyện. 
    # Nó cũng được lấy từ danh sách trainable_unet_numbers.
    last_trainable_unet = trainable_unet_numbers[-1]
    def move_unets(unet_training_mask):
        for i in range(len(decoder.unets)):
            if not unet_training_mask[i]:
                # Replace the unet from the module list with a nn.Identity(). This training script never uses unets that aren't being trained so this is fine.
                # Thay thế unet trong danh sách module bằng nn.Identity().
                # Kịch bản huấn luyện này không sử dụng các unet không được huấn luyện nên việc này là hợp lý.
                decoder.unets[i] = nn.Identity().to(inference_device)
    # Remove non-trainable unets
    # # Loại bỏ các unet không được huấn luyện
    move_unets(unet_training_mask) 

    # đinh nghĩa một Trainer bằng hàm decoder Trainer 
    trainer = DecoderTrainer(
        decoder=decoder,
        accelerator=accelerator,
        dataloaders=dataloaders,
        **kwargs
    )

    # Set up starting model and parameters based on a recalled state dict
    # start_epoch: Biến này lưu trữ epoch bắt đầu huấn luyện.
    start_epoch = 0
    # validation_losses: Danh sách lưu trữ các giá trị validation loss trong quá trình huấn luyện
    validation_losses = []
    # next_task: Xác định task tiếp theo sau khi gọi lại mô hình (có thể là “train” hoặc “val”).
    next_task = 'train'
    # sample: Số lượng mẫu đã thấy trong quá trình huấn luyện.
    sample = 0
    # samples_seen: Tổng số mẫu đã thấy trong quá trình huấn luyện.
    samples_seen = 0
    # val_sample: Số lượng mẫu kiểm tra đã thấy.
    val_sample = 0
    # step: Hàm lambda để tính số bước huấn luyện cho unet đầu tiên có thể huấn luyện.
    step = lambda: int(trainer.num_steps_taken(unet_number=first_trainable_unet))

    # if tracker.can_recall:: Điều kiện kiểm tra xem có thể gọi lại mô hình từ trạng thái trước đó hay không. Nếu điều kiện này đúng, 
    # chương trình sẽ thực hiện các bước tiếp theo.
    if tracker.can_recall:
        # : Gọi hàm recall_trainer để gọi lại thông tin về mô hình huấn luyện từ trạng thái trước đó. Các biến sau đó sẽ được cập nhật với thông tin đã gọi lại:
        start_epoch, validation_losses, next_task, recalled_sample, samples_seen = recall_trainer(tracker, trainer)
        # if next_task == 'train': sample = recalled_sample: Nếu task tiếp theo là “train”, 
        # thì cập nhật giá trị của sample với recalled_sample.
        if next_task == 'train':
            sample = recalled_sample
        # if next_task == 'val': val_sample = recalled_sample: Nếu task tiếp theo là “val”, thì cập nhật giá trị của val_sample với recalled_sample.
        if next_task == 'val':
            val_sample = recalled_sample
        # accelerator.print(...): In ra thông tin về việc gọi lại mô hình, số lượng mẫu đã thấy và giá trị validation loss tối thiểu.
        accelerator.print(f"Loaded model from {type(tracker.loader).__name__} on epoch {start_epoch} having seen {samples_seen} samples with minimum validation loss {min(validation_losses) if len(validation_losses) > 0 else 'N/A'}")
        accelerator.print(f"Starting training from task {next_task} at sample {sample} and validation sample {val_sample}")
    # trainer.to(device=inference_device): Di chuyển mô hình huấn luyện sang thiết bị inference_device (ví dụ: CPU hoặc GPU) để thực hiện dự đoán
    trainer.to(device=inference_device)

    # accelerator.print(print_ribbon("Generating Example Data", repeat=40)): In ra một dòng thông báo với tiêu đề “Generating Example Data” được trang trí bằng dấu gạch ngang.
    accelerator.print(print_ribbon("Generating Example Data", repeat=40))
    # accelerator.print("Việc này có thể mất một thời gian để tải danh sách các shard..."): Thông báo cho người dùng biết rằng việc tạo dữ liệu ví dụ có thể mất thời gian.
    accelerator.print("This can take a while to load the shard lists...")
    # if is_master:: Kiểm tra xem tiến trình hiện tại có phải là tiến trình chính hay không (ví dụ: tiến trình đầu tiên trong multi-GPU).
    if is_master:
        #  Tạo dữ liệu ví dụ cho huấn luyện từ tập dữ liệu huấn luyện.
        train_example_data = get_example_data(dataloaders["train_sampling"], inference_device, n_sample_images)
        accelerator.print("Generated training examples")
        # : Tạo dữ liệu ví dụ cho kiểm tra từ tập dữ liệu kiểm tra.
        test_example_data = get_example_data(dataloaders["test_sampling"], inference_device, n_sample_images)
        accelerator.print("Generated testing examples")
    
    # : Định nghĩa hàm send_to_device để chuyển dữ liệu sang thiết bị inference_device (ví dụ: CPU hoặc GPU).
    send_to_device = lambda arr: [x.to(device=inference_device, dtype=torch.float) for x in arr]

    # sample_length_tensor: Khởi tạo tensor có giá trị 0 và kiểu dữ liệu integer trên thiết bị inference_device.
    sample_length_tensor = torch.zeros(1, dtype=torch.int, device=inference_device)
    # Khởi tạo tensor có kích thước TRAIN_CALC_LOSS_EVERY_ITERS x trainer.num_unets và kiểu dữ liệu float trên thiết bị inference_device.
    unet_losses_tensor = torch.zeros(TRAIN_CALC_LOSS_EVERY_ITERS, trainer.num_unets, dtype=torch.float, device=inference_device)
    # Vòng lặp huấn luyện sẽ tiếp tục từ start_epoch đến epochs.
    for epoch in range(start_epoch, epochs):
        #accelerator.print(print_ribbon(f"Bắt đầu epoch {epoch}", repeat=40)): In ra một dòng thông báo với tiêu đề “Bắt đầu epoch” và số epoch tương ứng.
        accelerator.print(print_ribbon(f"Starting epoch {epoch}", repeat=40))

        # timer = Timer(): Khởi tạo đối tượng Timer để đo thời gian.
        timer = Timer()
        # last_sample = sample: Lưu giá trị của sample vào biến last_sample.
        last_sample = sample
        # last_snapshot = sample: Lưu giá trị của sample vào biến last_snapshot.
        last_snapshot = sample

        # if next_task == 'train':: Kiểm tra xem task tiếp theo sau khi gọi lại mô hình có phải là “train” hay không.
        if next_task == 'train':
            # Vòng lặp for i, (img, emb, txt) in enumerate(dataloaders["train"]):: Vòng lặp qua tập dữ liệu huấn luyện.
            for i, (img, emb, txt) in enumerate(dataloaders["train"]):
                # We want to count the total number of samples across all processes
                # sample_length_tensor[0] = len(img): Cập nhật giá trị của sample_length_tensor với độ dài của img.
                sample_length_tensor[0] = len(img)
                # Gom nhóm các giá trị sample_length_tensor từ tất cả các tiến trình.
                all_samples = accelerator.gather(sample_length_tensor)  # TODO: accelerator.reduce is broken when this was written. If it is fixed replace this.
                # Tính tổng số mẫu đã thấy.
                total_samples = all_samples.sum().item()
                # sample += total_samples: Cập nhật giá trị của sample bằng tổng số mẫu đã thấy.
                sample += total_samples
                # samples_seen += total_samples: Cập nhật giá trị của samples_seen bằng tổng số mẫu đã thấy.
                samples_seen += total_samples
                # img_emb = emb.get('img'): Lấy dữ liệu nhúng (embedding) của hình ảnh.
                img_emb = emb.get('img')
                # has_img_embedding = img_emb is not None: Kiểm tra xem có dữ liệu nhúng của hình ảnh hay không.
                has_img_embedding = img_emb is not None
                # if has_text_embedding:: Kiểm tra xem có dữ liệu nhúng của văn bản hay không.
                if has_text_embedding:
                    # text_emb, = send_to_device((text_emb,)): Chuyển dữ liệu nhúng của văn bản sang thiết bị inference_device.
                    text_emb, = send_to_device((text_emb,))
                # img, = send_to_device((img,)): Chuyển dữ liệu hình ảnh sang thiết bị inference_device.
                img, = send_to_device((img,))

                # trainer.train(): Bắt đầu quá trình huấn luyện mô hình.
                trainer.train()
                # sử dụng vòng lặp for duyệt qua tất acr các unet trong model 
                for unet in range(1, trainer.num_unets + 1):
                    # Check if this is a unet we are training 
                    # kiểm tra xem unet hiện tại có được huấn luyện hay không dựa trên unet_mask 
                    if not unet_training_mask[unet-1]: # Unet index is the unet number - 1
                        continue

                    # Xây dựng từ điểnn forward_params để lưu trữ các tham số trong 
                    # quá trình forward pass 
                    forward_params = {}

                    # KIỂM TRA XEM CÓ DƯ LIỆU NHÚNG CỦA HÌNH ẢNH HAY KHÔNG 
                    if has_img_embedding:
                        # nếu có dữ liêu nhúng hình ảnh gán giá trj của img_embed
                        # cho khóa image_embe trong forward_params
                        forward_params['image_embed'] = img_emb
                    # Nếu không có dữ liệu nhúng hình ảnh 
                    else:
                        # Forward pass automatically generates embedding
                        # đam bảo rằng clip không phải là None
                        assert clip is not None
                        # Thực hiện forward pass để tạo dữ liệu nhúng của hình ảnh (img) bằng mô hình clip
                        img_embed, img_encoding = clip.embed_image(img)
                        # Gán giá trị của img_embed cho khóa 'image_embed' trong forward_params.
                        forward_params['image_embed'] = img_embed
                    
                    # Kiểm tra xem văn có điều kiện hóa văn bản hay không 
                    if condition_on_text_encodings: 
                        # Và kiểm tra xem có nhúng văn bản hay không 
                        if has_img_embedding: 
                            # Sau đó thêm text_encoding và từ điển forward_params 
                            forward_params['text_encodings'] = text_emb 

                        # Nếu như không có nhúng văn bản 
                        else: 
                            # Sau đó chúng ta cần  chuyển văn bản thay thế 
                            # Đảm bảo rằng clip có tồn tại 
                            assert clip is not None 
                            # Thực hiện mã hóa token_bpe cho văn bản 
                            tokenized_texts = tokenize(txt, truncate=True).to(inference_device)
                            # Đảm bảo danh sách token_texts bằng với độ dài danh sách img 
                            assert tokenized_texts.shape[0] == len(img), f"The number of texts ({tokenized_texts.shape[0]}) should be the same as the number of images ({len(img)})"
                            # Sử dụng mô hình clip để nhúng các tiêu đề văn bản đồng thời học các mã hóa văn bản 
                            text_embed, text_eoncodings = clip.embed_text(tokenized_texts)

                            # Thêm các mã hóa văn bản vào từ điển forward_params 
                            forward_params['text_encodings'] = text_eoncodings 
                        
                    # Tính toán loss cho trình huấn luyện và đưa vào các tham số cần thiết 
                    # img danh sách chứa các ảnh thực tế và từ điển forward_params được nén các tham số 
                    # unet_number số lượng unet cho mô hình...
                    loss = trainer.forward(img, **forward_params, unet_number=unet, _device=inference_device)
                    # sau đó cập nhật unet cho 
                    trainer.update(unet_number=unet)
                    # ưu giá trị loss vào tensor unet_losses_tensor tại vị trí tương ứng với unet hiện tại.
                    unet_losses_tensor[i % TRAIN_CALC_LOSS_EVERY_ITERS, unet-1] = loss  

                # Tính tốc độ huấn luyện dựa trên số lượng mẫu đã thấy trong khoảng thời gian timer.elapsed
                samples_per_sec = (sample - last_sample) / timer.elapsed()
                # Đặt lại đối tương timer để bắt đầu đếm thời gian từ đầu 
                timer.reset()
                # Lưu giá trị của sample vào biến last_sample
                last_sample = sample

                # Kiểm tra xem đã đến lượt tính loss hay chưa sau mỗi TRAIN_CALC_LOSS_EVERY_
                if i % TRAIN_CALC_LOSS_EVERY_ITERS == 0:
                    # We want to average losses across all processes
                    # Gom nhóm các giá trị loss từ tất cả các tiến trình 
                    unet_all_losses = accelerator.gather(unet_losses_tensor)
                    # Tạo một mask để laoij bỏ đi các gái trị bằng 0 
                    mask = unet_all_losses != 0
                    # Nhân ma trận mask với  ênt_all_loss và sau đings tính giá trị loss trung bình cho từng 
                    # unet dựa trên mask
                    unet_average_loss = (unet_all_losses * mask).sum(dim=0) / mask.sum(dim=0)
                    #  Tạo một từ điển (dictionary) chứa thông tin về loss của từng unet được huấn luyện.
                    loss_map = { f"Unet {index} Training Loss": loss.item() for index, loss in enumerate(unet_average_loss) if unet_training_mask[index] }

                    # gather decay rate on each UNet
                    # Tạo một từ điển (dictionary) chứa thông tin về tỷ lệ giảm dần (decay rate) của từng unet dựa trên mô hình EMA (Exponential Moving Average)
                    ema_decay_list = {f"Unet {index} EMA Decay": ema_unet.get_current_decay() for index, ema_unet in enumerate(trainer.ema_unets) if unet_training_mask[index]}

                    # Tạo một từ điển log_data để lưu trữ các tham số cần thiết 
                    log_data = {
                        "Epoch": epoch,
                        "Sample": sample,
                        "Step": i,
                        "Samples per second": samples_per_sec,
                        "Samples Seen": samples_seen,
                        **ema_decay_list,
                        **loss_map
                    }
                    
                    # Kiểm tra xem có đnag là tiến trình chính hay không
                    if is_master:
                        # Nếu có nghi một nhật ký log với trình theo dõi tracker và
                        # các thông tin của từ điển log_data 
                        tracker.log(log_data, step=step())
                
                # kiểm tra xem quá trình hiện tại có phải là quá trình chính (thường được sử dụng trong huấn luyện phân tán) và xem có đến lúc lưu một bản chụp của mô hình không.
                # Nếu bất kỳ điều kiện sau đây được thỏa mãn, bản chụp sẽ được lưu:
                # Số lượng mẫu đã xử lý kể từ lần lưu bản chụp trước đó vượt quá save_every_n_samples
                # Cờ save_immediately được đặt thành True, và đây là lần lặp đầu tiên (tức là i == 0).
                if is_master and (last_snapshot + save_every_n_samples < sample or (save_immediately and i == 0)):  # This will miss by some amount every time, but it's not a big deal... I hope
                    # It is difficult to gather this kind of info on the accelerator, so we have to do it on the master
                    # Dòng này in thông báo cho biết đang lưu một bản chụp.
                    print("Saving snapshot")
                    # Cập nhật giá trị của last_snapshot thành chỉ số mẫu hiện tại.
                    last_snapshot = sample
                    # We need to know where the model should be saved
                    # Gọi một hàm có tên là save_trainer với các đối số liên quan đến trạng thái huấn luyện. 
                    # Hàm này có lẽ lưu các tham số của mô hình và thông tin liên quan khác.
                    save_trainer(tracker, trainer, epoch, sample, next_task, validation_losses, samples_seen)
                    # Kiểm tra xem biến n_sample_images có tồn tại và có lớn hơn 0 không.
                    if exists(n_sample_images) and n_sample_images > 0:
                        # Nếu đúng, thực hiện các hành động sau:
                        # Đặt mô hình vào chế độ đánh giá (trainer.eval()).
                        trainer.eval()
                        # Tạo ra các mẫu lưới bằng cách sử dụng hàm generate_grid_samples.
                        train_images, train_captions = generate_grid_samples(trainer, train_example_data, clip, first_trainable_unet, last_trainable_unet, condition_on_text_encodings, cond_scale, inference_device, "Train: ")
                        # Ghi lại các hình ảnh và chú thích tương ứng bằng hàm tracker.log_images.
                        tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step())
                
                # Kiểm tra xem biến epoch_samples có khác None và chỉ số mẫu hiện tại có vượt quá hoặc bằng epoch_samples không.
                if epoch_samples is not None and sample >= epoch_samples:
                    # Nếu đúng, vòng lặp huấn luyện kết thúc cho epoch hiện tại.

                    break
            # Đặt giá trị của biến next_task thành 'val', cho biết tác vụ tiếp theo là kiểm tra đánh giá.
            next_task = 'val'
            # Đặt lại chỉ số mẫu thành 0 cho epoch tiếp theo.
            sample = 0

        # Khởi tạo biến all_average_val_losses với giá trị None. Biến này có thể được sử dụng để lưu trữ 
        # các giá trị trung bình của hàm mất mát trong quá trình kiểm tra đánh giá.
        all_average_val_losses = None
        # Kiểm tra xem tác vụ tiếp theo là kiểm tra đánh giá ('val').
        # Nếu đúng, thực hiện các hành động sau đây:
        if next_task == 'val':
            # Đặt mô hình vào chế độ đánh giá (không tính toán gradient).
            trainer.eval()
            # In ra thông báo bắt đầu quá trình kiểm tra đánh giá cho epoch hiện tại.
            # Hàm print_ribbon có thể tạo ra một dòng kẻ ngang để làm đẹp cho thông báo.
            accelerator.print(print_ribbon(f"Starting Validation {epoch}", repeat=40))
            # Cập nhật giá trị của last_val_sample thành chỉ số mẫu kiểm tra đánh giá hiện tại.
            last_val_sample = val_sample
            # Khởi tạo một tensor có kích thước 1x1 với kiểu dữ liệu là số nguyên (dtype=torch.int) và thiết bị tính toán là inference_device.
            # Biến này có thể được sử dụng để theo dõi số lượng mẫu trong quá trình kiểm tra đánh giá.
            val_sample_length_tensor = torch.zeros(1, dtype=torch.int, device=inference_device)
            # Khởi tạo một tensor có kích thước 1xtrainer.num_unets với kiểu dữ liệu là số thực (dtype=torch.float) và thiết bị tính toán là inference_device.
            # Biến này có thể được sử dụng để tính toán giá trị trung bình của hàm mất mát trong quá trình kiểm tra đánh giá.
            average_val_loss_tensor = torch.zeros(1, trainer.num_unets, dtype=torch.float, device=inference_device)
            # Khởi tạo một đối tượng Timer để đo thời gian thực hiện quá trình kiểm tra đánh giá.
            timer = Timer()
            # Đợi cho tất cả các quá trình tính toán hoàn thành trước khi tiếp tục.
            # Thường được sử dụng trong huấn luyện phân tán để đồng bộ hóa các quá trình.
            accelerator.wait_for_everyone()
            # Đặt lại chỉ số i thành 0 để bắt đầu quá trình kiểm tra đánh giá.
            i = 0
            # Đây là một vòng lặp for để duyệt qua dữ liệu kiểm tra đánh giá ('val') trong dataloade
            # Mỗi lần lặp, biến i được cập nhật và các giá trị img, emb, và txt được gán từ dữ liệu tương ứng.
            for i, (img, emb, txt) in enumerate(dataloaders['val']):  # Use the accelerate prepared loader
                # Cập nhật giá trị của val_sample_length_tensor thành độ dài của img
                # val_sample_length_tensor có thể được sử dụng để theo dõi số lượng mẫu trong quá trình kiểm tra đánh giá.
                val_sample_length_tensor[0] = len(img)
                # Gom lại các giá trị val_sample_length_tensor từ tất cả các quá trình tính toán.
                # Điều này thường được sử dụng trong huấn luyện phân tán để đồng bộ hóa thông tin giữa các quá trình.
                all_samples = accelerator.gather(val_sample_length_tensor)
                # Tính tổng số lượng mẫu từ all_samples.
                # Giá trị này được sử dụng để cập nhật val_sample, chỉ số mẫu kiểm tra đánh giá.
                total_samples = all_samples.sum().item()
                val_sample += total_samples
                # Lấy giá trị của khóa 'img' từ emb. Biến img_emb có thể chứa nhúng hình ảnh (nếu có).
                img_emb = emb.get('img')
                # Kiểm tra xem có nhúng hình ảnh (img_emb) không.
                # Nếu có, biến has_img_embedding được đặt thành
                has_img_embedding = img_emb is not None
                # Nếu có nhúng hình ảnh, thực hiện các hành động sau:
                # Đưa img_emb vào thiết bị tính toán (sử dụng send_to_device).
                if has_img_embedding:
                    img_emb, = send_to_device((img_emb,))
                # Lấy giá trị của khóa 'text' từ emb.
                # Biến text_emb có thể chứa nhúng văn bản (nếu có).
                text_emb = emb.get('text')
                # # Kiểm tra xem có nhúng văn bản (text_emb) không.
                # Nếu có, biến has_text_embedding được đặt thành True.
                has_text_embedding = text_emb is not None
                # Nếu có nhúng văn bản, thực hiện các hành động sau:
                # Đưa text_emb vào thiết bị tính toán (sử dụng send_to_device).
                if has_text_embedding:
                    text_emb, = send_to_device((text_emb,))
                # Đưa img vào thiết bị tính toán (sử dụng send_to_device).
                img, = send_to_device((img,))

                # Lặp qua một vòng lặp từ 1-> Unets + 1 
                # là một danh sách các mạng unet trong decoder.unet
                for unet in range(1, len(decoder.unets)+1):
                    # Kiểm tra xem mạng unet hiện tại có cần huấn luyện không 
                    if not unet_training_mask[unet-1]: # Unet index is the unet number - 1
                        # No need to evaluate an unchanging unet
                        continue
                        
                    # Khời tạo một từ điển params đểm lưu trữ các thông số trong quá trình huấn luyện
                    forward_params = {}
                    # kiểm tra xem có tồn tại các nhúng hình ảnh không
                    if has_img_embedding:
                        # Nếu có thì chuyển các nhúng hình ảnh thành float datatype
                        # sau đó thêm chúng vào từ điển forwrad_params với key = image_embed 
                        forward_params['image_embed'] = img_emb.float()
                    # Nếu không tồn tại các nhúng hình ảnh có sẵn
                    else:
                        # Forward pass automatically generates embedding
                        # Chuyển qua việc tự động sinh ra các nhúng 
                        # 1: Đảm bảo rằng mô hình CLIP có tồn tại
                        assert clip is not None
                        # Thực hiện nhúng và mã hóa hình ảnh thông qua hàm clip.embed_image
                        img_embed, img_encoding = clip.embed_image(img)
                        # 3: Thêm các nhúng hình ảnh vào từ điển forward params 
                        # với key = image_embed 
                        forward_params['image_embed'] = img_embed
                    # Kiểm tra xem có điều kiện hóa văn bản không
                    if condition_on_text_encodings:
                        # Tiếp tục kiểm tra xem đã có sẵn các biểu diễn nhúng chú thích tiêu đề văn 
                        # bản hay chưa
                        if has_text_embedding:
                            # Nếu đã tồn tại sẵn biểu diễn này chuyển đổi nó sang float data_type 
                            # và thêm chúng vào từ điển forward_params với key = Text_encodings
                            forward_params['text_encodings'] = text_emb.float()
                        # Trường hợp không tồn tại văn bản nhúng sẵn
                        else:
                            # Then we need to pass the text instead
                            # Sau đó chúng ta chuyển qua văn bản thay thế 
                            # Đảm bảo rằng mô hình clip có tồn tại 
                            assert clip is not None
                            # Thực hiên mã hóa bpe_tokenize cho văn bản sử dùng hàm truncate để chia văn bản thành các phần 
                            # đều nhau và chuyển việc tính toán cho thiết bị được sử dụng
                            tokenized_texts = tokenize(txt, truncate=True).to(device=inference_device)
                            # Đảm bảo rằng độ dài của danh sách văn bản được tokenize = số kượng các hình ảnh 
                            assert tokenized_texts.shape[0] == len(img), f"The number of texts ({tokenized_texts.shape[0]}) should be the same as the number of images ({len(img)})"
                            # Sử dụng hàm embed_text để học các nhúng văn bản và các mã hóa văn bản từ mô hình clip
                            text_embed, text_encodings = clip.embed_text(tokenized_texts)
                            # Thêm các mã hoa văn bản vào từ điển forward_params với key = text_encodings
                            forward_params['text_encodings'] = text_encodings
                    
                    # Áp dụng quy tắc lan chuyển tiến qua mạng unet hiện tại để tính toán Loss cho quá trình huấn luyện
                    loss = trainer.forward(img.float(), **forward_params, unet_number=unet, _device=inference_device)
                    # Cập nhật giá trị trung bình của hàm mất mát cho mạng Unet hiện tại.
                    average_val_loss_tensor[0, unet-1] += loss

                # Kiểm tra xem đã đến lúc tính toán hàm mất mát cho quá trình kiểm
                # tra đánh giá hay chưa 
                if i % VALID_CALC_LOSS_EVERY_ITERS == 0:
                    # Tính toán tiics độ xử lý mẫu trong quá trình kiểm tra đánh giá 
                    # timer.elapsed() là thời gian đã trôi qua kể từ lần kiểm tra đánh giá trước đến hiện tại.
                    samples_per_sec = (val_sample - last_val_sample) / timer.elapsed()
                    # Đăt lại thời gian 
                    timer.reset()
                    # Cập nhật lại last_val_sample là số lượng mẫu đã được xử lý đến thời điển hiện tại 
                    last_val_sample = val_sample
                    # In ra thông báo về tiến trình kiểm tra đánh giá, bao gồm số lượng mẫu đã xử lý và tốc độ xử lý.
                    accelerator.print(f"Epoch {epoch}/{epochs} Val Step {i} -  Sample {val_sample} - {samples_per_sec:.2f} samples/sec")
                    # In ra giá trị trung bình của hàm mất mát tính đến lần kiểm tra đánh giá hiện tại.
                    accelerator.print(f"Loss: {(average_val_loss_tensor / (i+1))}")
                    accelerator.print("")
                
                # Kiểm tra xem đã đủ số lượng mẫu kiểm tra đánh giá hay chưa. 
                if validation_samples is not None and val_sample >= validation_samples:
                    # Nếu không đủ thoát khỏi vòng lặp 
                    break
            
            # In ra thông báo cho biết quá trình kiểm tra đánh giá đã hoàn thành sau i bước.
            print(f"Rank {accelerator.state.process_index} finished validation after {i} steps")
            # Đợi cho tất cả các quá trình hoàn thành tính toán trước khi tiếp tục 
            # Thường được sử dụng trong huấn luyện phân tán để đồng bộ hóa thông tin giữa các quá trình.
            accelerator.wait_for_everyone()
            # Chia tổng hàm mất mát cho số lượng bước kiểm tra đánh giá hiện tại để tính giá trị trung bình.
            average_val_loss_tensor /= i+1
            # Gather all the average loss tensors
            # Gom lại các giá trị hàm mất mát trung bình từ tất cả các quá trình tính toán.
            # Điều này thường được sử dụng trong huấn luyện phân tán để đồng bộ hóa thông tin giữa các quá trình.
            all_average_val_losses = accelerator.gather(average_val_loss_tensor)
            # Kiểm tra xem quá trình hiện tại có phải là quá trình chính 
            # (thường được sử dụng trong huấn luyện phân tán) hay không
            if is_master:
                # Nếu đung Tính giá trị trung bình của hàm mất mát cho từng mạng Unet cụ thể  
                unet_average_val_loss = all_average_val_losses.mean(dim=0)
                # Tạo một từ điển val_loss_map chứa thông tin về hàm mất mát kiểm tra đánh giá cho từng mạng Unet.
                # Chỉ bao gồm các mạng Unet có hàm mất mát khác 0.
                val_loss_map = { f"Unet {index} Validation Loss": loss.item() for index, loss in enumerate(unet_average_val_loss) if loss != 0 }
                # Ghi lại thông tin về hàm mất mát kiểm tra đánh giá vào hệ thống theo dõi (tracker).
                tracker.log(val_loss_map, step=step())
            # Đặt giá trị của next_task thành eval cho biết tác  vụ tiếp theo alf đánh giá 
            next_task = 'eval'

        # Kiểm tra xem có phải nhiênm vụ tiếp theo và đánh giá mô hình 
        if next_task == 'eval':
            # Nếu phải thì kiểm tra xem các tham số cấu hình cho việc đánh giá 
            # đã được thiết lập hay chưa
            if exists(evaluate_config):
                # In ra thông báo bắt đầu quá trình đánh giá cho epoch hiện tại.
                accelerator.print(print_ribbon(f"Starting Evaluation {epoch}", repeat=40))
                # Thực hiện đánh giá mô hình sử dụng hàm evaluate_trainer.
                evaluation = evaluate_trainer(trainer, dataloaders["val"], inference_device, first_trainable_unet, last_trainable_unet, clip=clip, inference_device=inference_device, **evaluate_config.model_dump(), condition_on_text_encodings=condition_on_text_encodings, cond_scale=cond_scale)
                # Nếu qúa trinh hiện tại đnag là quá trình chính 
                if is_master:
                    # Ghi lại kết quả đnah giá vào hệ thống theo dõi 
                    tracker.log(evaluation, step=step())
            
            # Câp nhật tác vụ tiếp theo là sample 
            next_task = 'sample'
            # Đặt số lượng mẫu = 0
            val_sample = 0


        # Kiểm tra xem nhiệm vụ tiếp theo có phải là sinh mâu không 
        if next_task == 'sample':
            # Và kiểm tra xem có đang ở qúa trình chính không 
            if is_master:
                # Nếu như đnag là quá trình chính tạo ra các ví dụ và lưu trữ mô hình 
                # Và thực hiện tạo các ảnh mẫu 
                
                # In ra thông báo đang thực hiện quá trình lấy mẫu 
                print(print_ribbon(f"Sampling Set {epoch}", repeat=40))
                # Sử dụng hàm generate_grid_samples để hiển thị các mẫu ảnh thật , ảnh giả dưới dạng lưới gridview 
                # Hàm này trả về 1 tuple gồm lưới grid co ảnh thật và ảnh giả cùng với captions 
                test_images, test_captions = generate_grid_samples(trainer, test_example_data, clip, first_trainable_unet, 
                                                last_trainable_unet, condition_on_text_encodings, cond_scale, inference_device, "Test: ")
                # Tương tự như trên ta áp dụng với dữ liệu train 
                train_images, train_captions = generate_grid_samples(trainer, train_example_data, clip, first_trainable_unet,
                                                last_trainable_unet, condition_on_text_encodings, cond_scale, inference_device, "Train: ")
                # Hàm log_image ghi lại hình ảnh và chú thích văn bản của các cặp hình ảnh test_image
                tracker.log_images(test_images, captions=test_captions, image_section="Test Samples", step=step())
                # Và cũng thực hiện việc tương tự với train_images
                tracker.log_images(train_images, captions=train_captions, image_section="Train Samples", step=step())

                # In ra thông tin cho biết bắt đầu quá trình lưu trữ
                print(print_ribbon(f"Starting Saving {epoch}", repeat=40))

                # Khởi tạo một biến is_bets  = False
                is_best = False
                # kiểm xe xem dnah sách chứa các giá trị trung bình los có tồn tại 
                if all_average_val_losses is not None:
                    # sau đó ta sử dụng nó để tính toán lại trung bình loss
                    average_loss = all_average_val_losses.mean(dim=0).sum() / sum(unet_training_mask)
                    # Kiểm tra xem hàm loss đã thỏa mãn kỳ vòng chưa
                    if len(validation_losses) == 0 or average_loss < min(validation_losses):
                        # nếu như thảo mãn đánh giấu cho mô hình đang ở trạng thái tôt nhất 
                        is_best = True
                    
                    # Thêm giá trị loss trung bình hiện tại vào danh sách validation loss
                    validation_losses.append(average_loss)
                # cuối cùng thực hiện lưu trữ mô tham số và mô hình tại kỷ nguyên hiện tại
                save_trainer(tracker, trainer, epoch, sample, next_task, validation_losses, samples_seen, is_best=is_best)
            
            # Đặt nhiệm vụ tiếp theo = train để thực hiện lại các kỹ nguyên 
            next_task = 'train'


# Xây dựng phương thức tracker , hàm này tạo ra một đối tượng Tracker để ghi lại thông tin về quá trình huấn luyện và lưu cấu hình
# Đây là một hàm có tên create_tracker nhận các tham số sau:
# accelerator: Đối tượng Accelerator.
# config: Cấu hình huấn luyện cho bộ giải mã.
# config_path: Đường dẫn đến tệp cấu hình.
# dummy: Cờ xác định xem chế độ giả lập (dummy) có được kích hoạt hay không (mặc định là False). 
# Hàm này trả về một đối tương tracker 
def create_tracker(accelerator: Accelerator, config: TrainDecoderConfig, config_path: str, dummy: bool = False) -> Tracker:
    # Gọi phương thức create của cấu hình tracker_config với các tham số tương ứng.
    tracker_config = config.tracker
    # tạo ra một từ điển accelerator_config chứa thông tin về loại phân tán, số lượng quá trình, và kiểu tính toán chính xác.
    accelerator_config = {
        # "Distributed": Được đặt thành True nếu loại phân tán khác với NO, ngược lại là False.
        "Distributed": accelerator.distributed_type != accelerate_dataclasses.DistributedType.NO,
        # "DistributedType": Lưu loại phân tán của accelerator.
        "DistributedType": accelerator.distributed_type,
        # "NumProcesses": Lưu số lượng quá trình.
        "NumProcesses": accelerator.num_processes,
        # "MixedPrecision": Lưu kiểu tính toán chính xác (mixed precision).
        "MixedPrecision": accelerator.mixed_precision
    }
    # accelerator.wait_for_everyone() đảm bảo rằng tất cả các quá trình tính toán đã hoàn thành trước khi tiếp tục.
    accelerator.wait_for_everyone()  # If nodes arrive at this point at different times they might try to autoresume the current run which makes no sense and will cause errors
    # Gọi phương thức create của cấu hình tracker_config với các tham số tương ứng.
    tracker: Tracker = tracker_config.create(config, accelerator_config, dummy_mode=dummy)
    # Lưu cấu hình vào tệp decoder_config.json.
    tracker.save_config(config_path, config_name='decoder_config.json')
    # Thêm thông tin về cấu hình mô hình vào đối tượng Tracker.
    tracker.add_save_metadata(state_dict_key='config', metadata=config.model_dump())
    return tracker


# Thiết lập phương thức initialize_training 
# phương thức này được sử dụng để thiết lập một môi 
# trường huấn luyên mô hình phân tán cho mô hình Decoder 
def initialize_training(config: TrainDecoderConfig, config_path):
    # Make sure if we are not loading, distributed models are initialized to the same values
    # Đặt seed chp PyTorch để đảm bảo rằng các khởi tạo ngẫu nhiên của mô hình phân tán có thể tái tạo 
    # đựoc và nhất quán trên tất cả các quá trình
    torch.manual_seed(config.seed)

    # Set up accelerator for configurable distributed training
    # Thiết lập một cấu trúc dữ liệu chứa các tham số cấu hình cho việc huấn luyện phân tán 
    # như việc tìm kiếm các tham số không được sử dụng (find_unused_parameters) và cấu hình đồ thị tĩnh (static_graph).
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config.train.find_unused_parameters, static_graph=config.train.static_graph)
    # Cấu trúc dữ liệu chứa các tham số khởi tạo nhóm quá trình phân tán bảo gồm thời gian chờ (timeout)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
    # Tạo một đối tượng acclerator với các cấu hình được định nghĩa. Giúp quản lý và tối ưu hóa việc huấn luyện trên nhiều GPU 
    # hoặc máy tính 
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])

    # iểm tra xem có nhiều hơn một quá trình (thường là GPU) không. Nếu có, 
    # thực hiện các bước để đảm bảo tất cả các quá trình có thể kết nối với nhau.
    if accelerator.num_processes > 1:
        # We are using distributed training and want to immediately ensure all can connect
        accelerator.print("Waiting for all processes to connect...")
        # Đợi cho đến khi tất cả các quá trình kết nối với nhau trước khi tiếp tục.
        accelerator.wait_for_everyone()
        accelerator.print("All processes online and connected")


    # If we are in deepspeed fp16 mode, we must ensure learned variance is off
    # Phần đầu tiên của mã kiểm tra xem mô hình đang chạy ở chế độ DeepSpeed fp16 và liệu cấu hình có chỉ định learned variance không.
    if accelerator.mixed_precision == "fp16" and accelerator.distributed_type == accelerate_dataclasses.DistributedType.DEEPSPEED and config.decoder.learned_variance:
        raise ValueError("DeepSpeed fp16 mode does not support learned variance")
    
    # Set up data
    # Nó tính toán số lượng shard cho mỗi quá trình dựa trên tổng số shard và số lượng quá trình (world size).
    all_shards = list(range(config.data.start_shard, config.data.end_shard + 1))
    world_size = accelerator.num_processes
    # Lấy ra chỉ số của tiến trình hiện tại 
    rank = accelerator.process_index
    # Tính toán số lượng shard cho mỗi tiến trình 
    shards_per_process = len(all_shards) // world_size
    assert shards_per_process > 0, "Not enough shards to split evenly"
    # Danh sách my_shards chứa các chỉ số shard được gán cho quá trình hiện tại (dựa trên chỉ số của nó).
    my_shards = all_shards[rank * shards_per_process: (rank + 1) * shards_per_process]

    # 
    # Các dataloader được tạo bằng cách sử dụng hàm create_dataloaders, nhận các tham số liên quan đến việc tải dữ liệu và tiền xử lý.
    dataloaders = create_dataloaders (
        available_shards=my_shards,
        img_preproc = config.data.img_preproc,
        train_prop = config.data.splits.train,
        val_prop = config.data.splits.val,
        test_prop = config.data.splits.test,
        n_sample_images=config.train.n_sample_images,
        **config.data.model_dump(),
        rank = rank,
        seed = config.seed,
    )

    # If clip is in the model, we need to remove it for compatibility with deepspeed
    # Khởi tạo mô hình clip = None
    clip = None
    # Kiểm tra xem mô hình Decoder có sử dụng clip không 
    if config.decoder.clip is not None:
        # Nêu có tạo một biến clip và gán giá trị của config.decoder.clip.create() cho nó. 
        # Điều này đảm bảo rằng chúng ta vẫn giữ lại clip để sử dụng trong quá trình huấn luyện, nhưng không sử dụng nó trong bộ giải mã vì gây ra vấn đề.
        clip = config.decoder.clip.create()  # Of course we keep it to use it during training, just not in the decoder as that causes issues
        # Sau đó, gán config.decoder.clip = None để loại bỏ clip khỏi bộ giải mã.
        config.decoder.clip = None
    # Create the decoder model and print basic info
    decoder = config.decoder.create()
    # Hàm get_num_parameters tính tổng số lượng tham số của mô hình (chỉ tính tham số có requires_grad hoặc không chỉ định only_training).
    get_num_parameters = lambda model, only_training=False: sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_training))

    # Create and initialize the tracker if we are the master
    tracker = create_tracker(accelerator, config, config_path, dummy = rank!=0)

    # Biến has_img_embeddings kiểm tra xem có sử dụng embeddings cho ảnh không (dựa trên URL của embeddings ảnh).
    has_img_embeddings = config.data.img_embeddings_url is not None
    # Biến has_text_embeddings kiểm tra xem có sử dụng embeddings cho văn bản không (dựa trên URL của embeddings văn bản).
    has_text_embeddings = config.data.text_embeddings_url is not None
    # Biến conditioning_on_text kiểm tra xem có sử dụng văn bản để điều kiện cho mô hình không (dựa trên cấu hình của các unet trong bộ giải mã).
    conditioning_on_text = any([unet.cond_on_text_encodings for unet in config.decoder.unets])

    # Biến has_clip_model kiểm tra xem có sử dụng mô hình Clip không (dựa trên việc clip có giá trị kh
    has_clip_model = clip is not None
    # Biến data_source_string được khởi tạo rỗng.
    data_source_string = ""


    # Mã kiểm tra xem có sử dụng embeddings ảnh được tính toán trước không (has_img_embeddings).
    if has_img_embeddings:
        # Nếu có, chuỗi data_source_string sẽ được thêm vào “precomputed image embeddings”.
        data_source_string += "precomputed image embeddings"
    # Nếu không, nhưng mô hình Clip có sẵn (has_clip_model), chuỗi sẽ được thêm vào “clip image embeddings generation”.
    elif has_clip_model:
        data_source_string += "clip image embeddings generation"
    # Nếu không có nguồn embeddings ảnh nào được chỉ định, một ValueError sẽ được ném ra.
    else:
        raise ValueError("No image embeddings source specified")
    # Nếu mô hình được điều kiện bởi văn bản (conditioning_on_text):
    if conditioning_on_text:
        # Kiểm tra xem có sử dụng embeddings văn bản được tính toán trước không (has_text_embeddings).
        if has_text_embeddings:
            # Nếu có, chuỗi data_source_string sẽ được thêm vào " and precomputed text embeddings".
            data_source_string += " and precomputed text embeddings"
        # Nếu không, nhưng mô hình Clip có sẵn (has_clip_model), chuỗi sẽ được thêm vào " and clip text encoding generation".
        elif has_clip_model:
            data_source_string += " and clip text encoding generation"
        # Nếu không có nguồn embeddings văn bản nào được chỉ định, một ValueError sẽ được ném ra.
        else:
            raise ValueError("No text embeddings source specified")

    # In Thông Tin Cấu Hình và Huấn Luyện
    # Mã sử dụng hàm accelerator.print để in thông tin cấu hình đã tải, số lượng quá trình và loại phân tán được sử dụng trong huấn luyện, 
    # cũng như thông tin về việc sử dụng nguồn dữ liệu nào và liệu có điều kiện bởi văn bản hay không.
    accelerator.print(print_ribbon("Loaded Config", repeat=40))
    accelerator.print(f"Running training with {accelerator.num_processes} processes and {accelerator.distributed_type} distributed training")
    accelerator.print(f"Training using {data_source_string}. {'conditioned on text' if conditioning_on_text else 'not conditioned on text'}")
    # Nó cũng in ra tổng số lượng tham số của mô hình giải mã và số lượng tham số đang được huấn luyện.
    accelerator.print(f"Number of parameters: {get_num_parameters(decoder)} total; {get_num_parameters(decoder, only_training=True)} training")
    # Cuối cùng, mã lặp qua từng unet trong mô hình giải mã và in ra số lượng tham số cho mỗi unet.
    for i, unet in enumerate(decoder.unets):
        accelerator.print(f"Unet {i} has {get_num_parameters(unet)} total; {get_num_parameters(unet, only_training=True)} training")

    # Gọi hàm train để huấn luyện mô hình giải mã.
    train(dataloaders, decoder, accelerator,
        clip=clip,
        tracker=tracker,
        inference_device=accelerator.device,
        evaluate_config=config.evaluate,
        condition_on_text_encodings=conditioning_on_text,
        **config.train.model_dump(),
    )
    
# Create a simple click command line interface to load the config and start the training
# Đoạn mã tạo một giao diện dòng lệnh đơn giản để tải cấu hình và bắt đầu huấn luyện.
@click.command()
# Tham số --config_file cho phép người dùng chỉ định tệp cấu hình.
@click.option("--config_file", default="./train_decoder_config.json", help="Path to config file")
def main(config_file):
    config_file_path = Path(config_file)
    config = TrainDecoderConfig.from_json_path(str(config_file_path))
    initialize_training(config, config_path=config_file_path)

if __name__ == "__main__":
    main()



