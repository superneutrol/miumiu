import json 
from torchvision import transforms as T 
from pydantic import BaseModel, validator, model_validator
from typing import List, Optional, Union, Tuple, Dict, Any, TypeVar

from x_clip import CLIP as XCLIP
from open_clip import list_pretrained
from coca_pytorch import CoCa

from dalle2_pytorch import (
    CoCaAdapter, 
    OpenAIClipAdapter,
    OpenClipAdapter,
    Unet,
    Decoder,
    DiffusionPrior,
    DiffusionPriorNetwork,
    XClipAdapter
)
from dalle2_pytorch.trackers import Tracker, create_loader, create_logger, create_saver


# Xây dựng các hàm chức năng 
# 1 exits trả về chính nó nếu nó có tồn tại 
def exists(val):
    return val is not None 

# và default sẽ trả về giá trị 
# là chính nó thông qua exists hoặc một giá trị d nếu nó không tồn tại 
def default(val, d):
    return val if exists(val) else d 

# tạo một biến kiểu Type variable có tên là InnerType biến này có thể được sử dụng 
# để chỉ định một kiểu cụ thể nào đó sẽ đươc xác định sau khi kiêủ này 
# được sử dụng trong một hàm hoặc lớp 
InnerType = TypeVar['InnerType']

# Tạo một kiển mới có tên là ListOrTyple được định nghĩa là một union type của lIst và Tuple 
# cả 2 đều chứa các phần tử kiểu InnerType. có nghĩa biến của kiểu listortuple cps thể là 
# một trong 2 trường hợp 
ListOrTuple = Union[List[InnerType], Tuple[InnerType]]
# Tạo một kiểu mới có tên là SingularOrIterable, được định nghĩa là một union type của InnerType và ListOrTuple[InnerType].
#  Điều này có nghĩa là biến của kiểu SingularOrIterable có thể là một giá trị đơn lẻ 
# của kiểu InnerType hoặc một iterable (list hoặc tuple) chứa các phần tử của kiểu InnerType.
SingularOrIterable = Union[InnerType, ListOrTuple[InnerType]]


# Thiết lập lớp TrainSplitConfig lớp này đại diện cho cấu hình chia dữ liệu 
# thành các tập huấn luyện, kiểm định, và kiểm tra 
# Nó kế thừa từ BaseModel 
class TrainSplitConfig(BaseModel):
    # định nghĩa 3 thuộc tính float 
    train: float = 0.75
    val: float = 0.15
    test: float = 0.1

    @model_validator(mode = 'after') # chỉ ra rằng phương thức tiếp theo 
    # là một hàm kiểm tra hợp lệ 
    def validate_all(self, m):
        # tính tổng thực tế của 3 tỷ lệ  train, val, test 
        actual_sum = sum([*dict(self).values()])
        # NẾU NHƯ TỔNG THỰC TẾ != 1 
        if actual_sum != 1.:
            # ném ra một lỗi values error 
            raise ValueError(f'{dict(self).keys()} must sum to 1.0. Found: {actual_sum}')
        # trả về self thể hiện của (TrainSplitConfig) nếu kiểm tra hợp lệ thành công 
        return self

# xây dựng lớp TrackerLogConfig lớp này đại diện cho cấu hình ghi nhật ký 
# Nó cũng được kế thừa từ BaseModel 
class TrackerLogConfig(BaseModel):
    # định nghĩa thuộc tính log_type một chuỗi console chỉ ra loại nhật ký 
    log_type: str = 'console'
    # resume: Một giá trị boolean với giá trị mặc định là False, dùng cho nhật ký
    #  được lưu ở vị trí độc đáo (để tiếp tục một lần chạy trước đó).
    resume: bool = False  # For logs that are saved to unique locations, resume a previous run
    # auto_resume: Một giá trị boolean với giá trị mặc định là False,
    # dùng để tiếp tục từ một lần chạy bị lỗi nếu quá trình khởi động lại.
    auto_resume: bool = False  # If the process crashes and restarts, resume from the run that crashed
    # Một giá trị boolean với giá trị mặc định là False, kiểm soát mức độ chi tiết của nhật ký.
    verbose: bool = False

    class Config:
        # Each individual log type has it's own arguments that will be passed through the config
        extra = "allow"

    # phương thức create nhận một đối số data_path 
    def create(self, data_path: str):
        # tạo một từ điển các đối số 
        kwargs = self.dict()
        # gọi hàm create_logger với loại nhật ký được chỉ định, đường dẫn dữ liệu và accs đối số 
        # từ khóa bổ dung 
        return create_logger(self.log_type, data_path, **kwargs)
    


# Xây dựng lớp TrackerLoadConfig đại diện cho một cấu hình tải dữ liệu 
# 
class TrackerLoadConfig(BaseModel):
    # load_form một biến tùy chọn kiểu chuỗi 
    load_from: Optional[str] = None
    # only_auto_resume chỉ định rằng việc tải chỉ được thực hiện nếu logger đang tự 
    # động khôi phục sau sự cố 
    only_auto_resume: bool = False  # Only attempt to load if the logger is auto-resuming

    # Một lớp nội tại config 
    class Config:
        # cho phéo thêm các trường không được định nghĩa rõ ràng trong lớp (extra = 'allow')
        extra = "allow"

    # xây dựng phương thức create
    def create(self, data_path: str):
        # tạo một từ điển cac đối số từ khóa kwargs từ các thuộc tính của thể hiện
        kwargs = self.dict()
        # nếu laod_form là None 
        if self.load_from is None:
            # phương thức sẽ trả về None
            return None
        # Nếu không gọi hàm create_loader với load_form, data_path và acc đối số từ 
        # kháo để tạo loader 
        return create_loader(self.load_from, data_path, **kwargs)

# Xây dựng lớp TrackerSaveConfig để thực hiện cấu hình lưu trữ dữ liệu
class TrackerSaveConfig(BaseModel):
    # save_to một chuỗi với giá trị ămcj định là local chỉ định vị trí lưu trữ dữ liệu 
    save_to: str = 'local'
    # save_all chỉ định có lưu tất cả các phiên bản hay khôngb
    save_all: bool = False
    # save_latest chỉ định có lưu phiên bản mới nhất hay không 
    save_latest: bool = True
    # và save_best chỉ định có lưu phiên bản tốt nhát hay không 
    save_best: bool = True

    # class config cho phép thêm các trường không được định nghĩa rõ ràng cho lớp
    class Config:
        extra = "allow"

    # phương thức create
    def create(self, data_path: str):
        # tạo một từ điển các đối số từ khóa từ các thuộc tính của thể hiện.
        kwargs = self.dict()
        # Gọi hàm create_saver với save_to, data_path, và các đối số từ khóa để tạo saver
        return create_saver(self.save_to, data_path, **kwargs)
    

# Xây dựng lớp TrackerConfig đại diện cho một cấu hình  chung của một tracker
# có thể hiểu là công cụ để theo dõi và quản lý dữ liệu hoặc trạng thái trong một ứng dụng
class TrackerConfig(BaseModel):
    # data_path một chuỗi chỉ định đường dẫn nơi lưu trữ tracker với giá trị mặc định là 
    # tracker_data
    data_path: str = '.tracker_data'
    # overwrite chỉ định xem liẹu có ghi đè dữ liệu cũ không 
    overwrite_data_path: bool = False
    # log thể hiện một lớp trackerConfig định nghĩa cấu h8nhf cho việc ghi nhật ký 
    log: TrackerLogConfig
    # một biến tùy chọn  của lớp TrackerLoadonfig định nghĩa cấu hình cho việc 
    # tải dữ liệu 
    load: Optional[TrackerLoadConfig] = None
    # một union type của List[TrackerSaveConfig] hoặc TrackerSaveConfig, định nghĩa cấu hình cho việc lưu dữ liệu.
    save: Union[List[TrackerSaveConfig], TrackerSaveConfig]


    # thiết lập phương thức create được khởi tạo là một thể hiên của tracker với các 
    # thông số được cung cấp 
    def create(self, full_config: BaseModel, extra_config: dict, dummy_mode: bool = False) -> Tracker:
        # tạo một thể hiện của lớp tracker 
        tracker = Tracker(self.data_path, dummy_mode=dummy_mode, overwrite_data_path=self.overwrite_data_path)
        # Add the logger
        # thêm một logger vào tracker dựa trên cấu hình log
        tracker.add_logger(self.log.create(self.data_path))
        # Add the loader 
        # nếu có cấu hình tải 
        if self.load is not None:
            # thêm một laoder và tracker 
            tracker.add_loader(self.load.create(self.data_path))
        # Add the saver or savers
        # kiểm tra xem save có phải là một danh sách hoặc cấu hình 
        # lưu trữ hay không 
        if isinstance(self.save, list):
            # nếu đúng duyệt qua qua danh sách các save 
            for save_config in self.save:
                # và thêm các add_saver và tracker với các save_config 
                tracker.add_saver(save_config.create(self.data_path))
        # nếu không 
        else:
            # thfi ta chỉ thêm một saver 
            tracker.add_saver(self.save.create(self.data_path))
        # Initialize all the components and verify that all data is valid
        # Khởi tạo tracker với cấu hình đầy đủ và cấu hình bổ sung, đồng thời kiểm tra tính hợp lệ của dữ liệu.
        tracker.init(full_config, extra_config)
        # Trả về thể hiện của Tracker đã được khởi tạo.
        return tracker#
    


# khuếch tán các lớp dyantic trước  
# Xây dựng lớp Adapterconfig lớp này kế thừa từ BaseModel và định nghĩa một 
# cấu hình cho việc tạo adapter 
class AdapterConfig(BaseModel):
    # định nghĩa nah phát hành của Adapter 
    make: str = "openai"
    # model chỉ định mô hình cụ thể được sử dụng
    model: str = "ViT-L/14"
    # Một từ điển tùy chọn chứa các tham số khởi tạo cho mô hình cơ sở 
    # có thể là None
    base_model_kwargs: Optional[Dict[str, Any]] = None

    # phưng thức create 
    def create(self):
        # kiểm tra giá trị của make 
        if self.make == "openai":
            # sau đó tạo một adapter tương ứng 
            return OpenAIClipAdapter(self.model)
        # nếu make là opne ai phương thức lấy danh sahcs các mô hình đã được 
        # huấn luyện trước pretrained  
        elif self.make == "open_clip":
            pretrained = dict(list_pretrained())
            checkpoint = pretrained[self.model]
            # tìm  checkpoint cho mô hình được chỉ định và trả về thể hiện của 
            # OpenClipAdapter 
            return OpenClipAdapter(name=self.model, pretrained=checkpoint)
    
        # nếu make là x-clip 
        elif self.make == "x-clip":
            # phương thức trả về một thể hiện của XClipAdapter với các tham số được truyền qua
            # base_model_kwargs
            return XClipAdapter(XCLIP(**self.base_model_kwargs))
        # Nếu make là "coca", phương thức trả về một thể hiện của CoCaAdapter 
        # với các tham số được truyền qua base_model_kwargs
        elif self.make == "coca":
            return CoCaAdapter(CoCa(**self.base_model_kwargs))
        # Nếu make không phải là một trong các giá trị trên, phương thức ném 
        # ra một AttributeError với thông báo không có adapter nào có tên đó.
        else:
            raise AttributeError("No adapter with that name is available.")


# xây dựng lớp DiffusionPriorNetworkConfig được sử dụng để cấu hình một 
# mạng lưới ưu tiên 
class DiffusionPriorNetworkConfig(BaseModel):
    dim: int
    depth: int
    max_text_len: Optional[int] = None
    num_timesteps: Optional[int] = None
    # một số nguyên chỉ định số lượng nhúng thời gian 
    num_time_embeds: int = 1
    # một số nguyên chỉ định số lượng nhúng hình ảnh 
    num_image_embeds: int = 1
    # một số nguyên chỉ định số lượng nhúng văn bản 
    num_text_embeds: int = 1
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    norm_in: bool = False
    # norm_out chỉ định xem có sử dụng chuẩn hóa đầu ra hay không 
    norm_out: bool = True
    attn_dropout: float = 0.
    # chỉ định tỷ lệ dropout trong feedforward network. 
    ff_dropout: float = 0.
    # chỉ định liệu có một lớp chiếu cuối cùng không.
    final_proj: bool = True
    # normform  chỉ định liệu sử dụng cấu trúc chuẩn hóa (normformer) không.
    normformer: bool = False
    # rotary_emb  chỉ định liệu sử dụng nhúng xoay (rotary embeddings) không.
    rotary_emb: bool = True

    # Lớp nội tại Config:
    # Cho phép thêm các trường không được định nghĩa rõ ràng trong lớp (extra = "allow").
    class Config:
        extra = "allow"

    def create(self):
        # tạo một từ điển các đối số từ khóa kwargs từ các thuộc tính của thể hiện 
        kwargs = self.dict()
        # trả về một thể hiện mới của DiffusionPriorNetWork với các tham số 
        # được truyền qua kwargs 
        return DiffusionPriorNetwork(**kwargs)




class DiffusionPriorConfig(BaseModel):
    clip: Optional[AdapterConfig] = None
    net: DiffusionPriorNetworkConfig
    image_embed_dim: int
    image_size: int
    image_channels: int = 3
    timesteps: int = 1000
    # sample_timestep chỉ định số bước thời gian cho việc láy mẫu 
    sample_timesteps: Optional[int] = None
    cond_drop_prob: float = 0.
    loss_type: str = 'l2'
    # predict_x_start chỉ định xem lệu có dự đoán điểm xuất phát hay không 
    predict_x_start: bool = True
    # một chuỗi chỉ định lịc trình beta
    beta_schedule: str = 'cosine'
    condition_on_text_encodings: bool = True

    # xây dựng một lớp config cho phép thêm các trường hợp không đưoc thể hiện rõ ràng 
    # trong lớp extra ='allow' 
    class Config:
        extra = "allow"

    def create(self):
        # xây dựng một từ điển các đối số từ khóa kwargs từ các thuộc tính được 
        # thể hiện 
        kwargs = self.dict()

        # kiểm tra xem trong từ điển có tồn tại một chuỗi clip và loại bỏ 
        # nó khỏi từ điển 
        has_clip = exists(kwargs.pop('clip'))
        # sau đó loại bỏ khóa net khỏi từ điển kwargs 
        kwargs.pop('net')

        clip = None # gán cho kháo clip = None
        if has_clip: # nếu như thuộc tính hss_clip tồn tại 
            # khởi tạo một clip khác 
            clip = self.clip.create()

        # khởi tạo một mạng khuếch tán 
        diffusion_prior_network = self.net.create()
        # Cuối cùng tạo một thể hiện của mangj lưới ưu tiên lan truyền với 
        # mô hình clip diffusion
        return DiffusionPrior(net = diffusion_prior_network, clip = clip, **kwargs)


# xây dụng lớp diffusion ... Cấu hình các tham số sử dụng cho cấu hình Train
class DiffusionPriorTrainConfig(BaseModel):
    epochs: int = 1
    lr: float = 1.1e-4
    wd: float = 6.02e-2
    max_grad_norm: float = 0.5
    use_ema: bool = True
    ema_beta: float = 0.99
    amp: bool = False
    warmup_steps: Optional[int] = None   # number of warmup steps
    save_every_seconds: int = 3600       # how often to save
    eval_timesteps: List[int] = [64]     # which sampling timesteps to evaluate with
    best_validation_loss: float = 1e9    # the current best valudation loss observed
    current_epoch: int = 0               # the current epoch
    num_samples_seen: int = 0            # the current number of samples seen
    random_seed: int = 0                 # manual seed for torch


# tương tự như trên lớp này thể hiện cho cấu hình của dữ liệu 
# được sử dụng để xây dựng vào thao tác xử lý 

class DiffusionPriorDataConfig(BaseModel):
    image_url: str                   # path to embeddings folder
    meta_url: str                    # path to metadata (captions) for images
    splits: TrainSplitConfig         # define train, validation, test splits for your dataset
    batch_size: int                  # per-gpu batch size used to train the model
    num_data_points: int = 25e7      # total number of datapoints to train on
    eval_every_seconds: int = 3600   # validation statistics will be performed this often


# Xây dựng cấu hình train cho diffusion Prior
class TrainDiffusionPriorConfig(BaseModel):
    # định nghĩa một câú hình pior diffusion
    prior: DiffusionPriorConfig
    # data config
    data: DiffusionPriorDataConfig
    # cấu hình tham số cho việc đào tạo mô hình
    train: DiffusionPriorTrainConfig
    # và trình theo dõi và gi các loại nhật ký
    tracker: TrackerConfig

    @classmethod # định nghĩa một phương thức lớp
    def from_json_path(cls, json_path):
        # mở các file jsom
        with open(json_path) as f:
            # sau đó đọc chúng 
            config = json.load(f)
        # và lưu chúng vào cấu hình config 
        return cls(**config)

# decoder pydantic classes

class UnetConfig(BaseModel):
    dim: int
    dim_mults: ListOrTuple[int]
    image_embed_dim: Optional[int] = None
    text_embed_dim: Optional[int] = None
    cond_on_text_encodings: Optional[bool] = None
    cond_dim: Optional[int] = None
    channels: int = 3
    self_attn: SingularOrIterable[bool] = False
    attn_dim_head: int = 32
    attn_heads: int = 16
    init_cross_embed: bool = True

    # Lớp nội tạng config 
    class Config: #
        # cho phép thêm các trường không được định nghĩa rõ ràng trong lớp 
        # (extra = 'allow')
        extra = "allow"



# Xây dụng lớp decodeConfig cấu hình cho các thuộc tính được sử dụng 
# cho việc giải mã  
class DecoderConfig(BaseModel):
    # Cấu hình một danh sách hoặc tuple chứa cấu hình Unetconfig cho một 
    # mạng Unet riêng lẻ 
    unets: ListOrTuple[UnetConfig]
    # image_size chỉ định kích thước đơn lẻ của hình ảnh
    image_size: Optional[int] = None
    # image_sizes một danh sách hoặc tuple chứa các giá trị mỗi giá trị thể định 
    # kích thước cho một hình ảnh khác nhau
    image_sizes: ListOrTuple[int] = None
    # clip một biến tuy chọn của lớp AdapterConfig chỉ định mô hình CLIP để sử dụng 
    # nếu nhúng không được cung cấp 
    clip: Optional[AdapterConfig] = None   # The clip model to use if embeddings are not provided
    channels: int = 3
    timesteps: int = 1000
    # sample_timestep số bươc trời gian tùy chọn cho việc lấy mẫu 
    sample_timesteps: Optional[SingularOrIterable[Optional[int]]] = None
    loss_type: str = 'l2'
    beta_schedule: Optional[ListOrTuple[str]] = None  # None means all cosine
    # một biến chỉ đih liệu có học các phương sai hay không 
    learned_variance: SingularOrIterable[bool] = True
    # image_cond_drop_prob: Xác suất dropout khi điều kiện hóa hình ảnh.
    image_cond_drop_prob: float = 0.1
    # text_cond_drop_prob: Xác suất dropout khi điều kiện hóa văn bản.
    text_cond_drop_prob: float = 0.5

    def create(self):
        # tạo một từ điển các đối sô từ khóa (decoder_kwargs) từ các thuộc tính 
        # của thể hiện
        decoder_kwargs = self.dict()

        # thực hiện loại bỏ cấu hình unet
        unet_configs = decoder_kwargs.pop('unets')
        # sau đó thực hiện tạo một danh sách các mạng Unet từ cấu hình này 
        unets = [Unet(**config) for config in unet_configs]

        # lọai bỏ khóa clip khỏi từ điển lưu trữ nếu nó tồn tại
        has_clip = exists(decoder_kwargs.pop('clip'))
        # sau đó gán cho clip = None
        clip = None
        # # kiểm tra xem có tồn tại cấu hình clip hay không và tạo ra một thể hiện của adapter tương ứng 
        # nếu có
        if has_clip:
            clip = self.clip.create()
        
        # trả về một thể hiện mới của Decoder với các mạng unet và adapter Clip nếu có 
        # cùng với các đối số từ kháo khác 
        return Decoder(unets, clip=clip, **decoder_kwargs)

    @validator('image_sizes') #bộ trang trí chỉ định răgf phương thức check_image sẽ được sử dụng 
    # để xác nhận trường image_sizes trong decoderConfig
    def check_image_sizes(cls, image_sizes, values):
        # kiểm tra xem chỉ có một trong 2 thuộc tính img_size hoặc img_sizes được 
        # cung câos hay không. toán tử ^ trả về true nếu 1 trong hai điều kiện đúng 
        if exists(values.get('image_size')) ^ exists(image_sizes):
            return image_sizes
        # nếu cả 2 không có thuộc tính nào được cung cấp ném ra một lỗi 
        raise ValueError('either image_size or image_sizes is required, but not both')

    # lớp Config cho phép thêm các trường không được định nghĩa rõ ràng 
    # trong lớp extra = 'allow' 
    class Config:
        extra = "allow"



# Xây dựng lớp decoderDataConfig lớp này đại diện cho cấu hình dữ liệu cho decoder 
# thường được sử dụng đểtaoj ra dữ liệu đầu vào cho mạng khuếch tán 
class DecoderDataConfig(BaseModel):
    # webdataset_base_url: Một chuỗi chỉ định đường dẫn đến một tập dữ liệu webdataset chứa hình ảnh JPG.
    webdataset_base_url: str                     # path to a webdataset with jpg images
    img_embeddings_url: Optional[str] = None     # path to .npy files with embeddings
    text_embeddings_url: Optional[str] = None    # path to .npy files with embeddings
    # num_worker chỉ định số lượng tiến trình để tải dữ liệu 
    num_workers: int = 4
    # kích thước mỗi lô 
    batch_size: int = 64
    # start_shard chỉ điinh phần shard bắt đầu 
    start_shard: int = 0
    # và phần shard kết thúc 
    end_shard: int = 9999999
    # chiều rộng của shard 
    shard_width: int = 6
    # index_width: Số nguyên chỉ định chiều rộng của chỉ số.
    index_width: int = 4
    splits: TrainSplitConfig
    shuffle_train: bool = True
    resample_train: bool = False
    # preprocessing: Một từ điển chứa các tham số cho việc tiền xử lý dữ liệu hình ảnh (ví dụ: chuyển đổi thành tensor).
    preprocessing: Dict[str, Any] = {'ToTensor': True}

    # định nghĩa một phươngthuwcs img_process 
    @property # nó sẽ được gọi như một thuộc tính của lớp
    def img_preproc(self): # Đây là một thuộc tính (property) của lớp, 
        # trả về một hàm biến đổi (transformation) cho việc tiền xử lý hình ảnh.
        def _get_transformation(transformation_name, **kwargs):
            # xác định loại biến đổi dựa trên tham số được truyền vào
            if transformation_name == "RandomResizedCrop":
                return T.RandomResizedCrop(**kwargs)
            elif transformation_name == "RandomHorizontalFlip":
                return T.RandomHorizontalFlip()
            elif transformation_name == "ToTensor":
                return T.ToTensor()

        transforms = []
        # duyệt qua dnah sách các biến đổi được chỉ đinhkj trong preprocessing vào tạo danh sách 
        # các biến đổi 
        for transform_name, transform_kwargs_or_bool in self.preprocessing.items():
            # trả về một từ điển nếu như transform_kwargs_or_bool  là một từ điển hoặc giá trị transform_kwargs_or_bool 
            transform_kwargs = {} if not isinstance(transform_kwargs_or_bool, dict) else transform_kwargs_or_bool
            transforms.append(_get_transformation(transform_name, **transform_kwargs))
        
        # rả về một hàm tổng hợp (compose) các biến đổi. 
        return T.Compose(transforms)


# xây dựng lớp DecoderTrainConfig lớp này đặc trung cấu hình huấn luyệnc của 
# decoder 
class DecoderTrainConfig(BaseModel):
    epochs: int = 20
    lr: SingularOrIterable[float] = 1e-4
    wd: SingularOrIterable[float] = 0.01
    # warup số bước khởi động trước khi bắt đầu quá trình tối ưu hóa 
    warmup_steps: Optional[SingularOrIterable[int]] = None
    # tìm kiếm các tham số không được sử dụng trong qúa trình huấn luyện 
    find_unused_parameters: bool = True
    # đồ thị tính toán tĩnh 
    static_graph: bool = True
    # max_grad_norm: Giới hạn lớn nhất của norm gradient để cắt tỉa gradient.
    max_grad_norm: SingularOrIterable[float] = 0.5
    # save_every_n_samples: Lưu mô hình sau mỗi số mẫu nhất định
    save_every_n_samples: int = 100000
    # n_sample_images: Số lượng hình ảnh mẫu được tạo ra khi lấy mẫu từ tập dữ liệu huấn luyện và kiểm tra.
    n_sample_images: int = 6                       # The number of example images to produce when sampling the train and test dataset
    # Hệ số điều kiện hóa
    cond_scale: Union[float, List[float]] = 1.0
    device: str = 'cuda:0'
    # epoch_samples: Giới hạn số mẫu mỗi kỳ huấn luyện.
    epoch_samples: Optional[int] = None                      # Limits the number of samples per epoch. None means no limit. Required if resample_train is true as otherwise the number of samples per epoch is infinite.
    # validation_samples: Tương tự như trên nhưng dành cho việc kiểm định.
    validation_samples: Optional[int] = None                 # Same as above but for validation.
    # save_immediately: Lưu ngay lập tức sau mỗi kỳ huấn luyện. 
    save_immediately: bool = False
    # use_ema: Sử dụng trung bình di động mũ (EMA) trong quá trình huấn luyện.
    use_ema: bool = True
    # ema_beta: Giá trị beta cho EMA 
    ema_beta: float = 0.999
    # : Sử dụng huấn luyện độ chính xác hỗn hợp tự động (AMP).
    amp: bool = False
    # unet_training_mask: Chọn lọc các mạng U-Net nào sẽ được sử dụng trong quá trình huấn luyện.
    unet_training_mask: Optional[ListOrTuple[bool]] = None   # If None, use all unets

class DecoderEvaluateConfig(BaseModel):
    # n_evaluation_samples: Số lượng mẫu được sử dụng cho việc đánh giá 
    n_evaluation_samples: int = 1000
    # FID, IS, KID, LPIPS: Các từ điển tùy chọn chứa các chỉ số đánh giá cụ thể như Fréchet Inception Distance, Inception Score, 
    # Kernel Inception Distance, Learned Perceptual Image Patch Similarity.
    FID: Optional[Dict[str, Any]] = None
    IS: Optional[Dict[str, Any]] = None
    KID: Optional[Dict[str, Any]] = None
    LPIPS: Optional[Dict[str, Any]] = None



# xây dựng TrainDecoderConfig:
# Lớp này kết hợp các cấu hình từ các lớp khác để tạo ra một cấu hình toàn diện cho việc huấn luyện decoder.
class TrainDecoderConfig(BaseModel):
    #decoder: Một thể hiện của lớp DecoderConfig, chứa cấu hình cho decoder.
    decoder: DecoderConfig
    # data: Một thể hiện của lớp DecoderDataConfig, chứa cấu hình cho dữ liệu đầu vào.
    data: DecoderDataConfig
    # train: Một thể hiện của lớp DecoderTrainConfig, chứa cấu hình cho quá trình huấn luyện.
    train: DecoderTrainConfig
    # evaluate: Một thể hiện của lớp DecoderEvaluateConfig, chứa cấu hình cho việc đánh giá mô hình.
    evaluate: DecoderEvaluateConfig
    # tracker: Một thể hiện của lớp TrackerConfig, chứa cấu hình cho việc theo dõi và ghi nhật ký quá trình huấn luyện.
    tracker: TrackerConfig
    # seed: Một số nguyên chỉ định giá trị seed cho việc khởi tạo ngẫu nhiên, giúp đảm bảo tính nhất quán trong các lần chạy khác nhau.
    seed: int = 0

    # xây dựng Phương thức from_json_path(cls, json_path):
    @classmethod
    # Đây là một phương thức lớp (classmethod) được sử dụng để tạo một thể hiện của TrainDecoderConfig từ một tệp JSON.
    def from_json_path(cls, json_path):
        # hương thức mở tệp JSON, đọc cấu hình và in ra nó. 
        with open(json_path) as f:
            # hương thức mở tệp JSON, đọc cấu hình và in ra nó
            config = json.load(f)
            print(config)
            # Sau đó, tạo và trả về một thể hiện mới của TrainDecoderConfig với cấu hình đã được đọc.
        return cls(**config)
    
    # @model_validator được sử dụng để đảm bảo rằng cấu hình cung cấp đủ thông tin để lấy các nhúng cần thiết cho quá trình huấn luyện. 
    # Dưới đây là giải thích chi tiết:
    # @model_validator(mode = 'after'): Bộ trang trí này chỉ định rằng phương thức check_has_embeddings sẽ được gọi
    # sau khi tất cả các trường khác của lớp đã được xác nhận.
    @model_validator(mode = 'after')
    def check_has_embeddings(self, m):
        # Makes sure that enough information is provided to get the embeddings specified for training
        # values = dict(self): Lấy tất cả các giá trị của thể hiện hiện tại và chuyển chúng thành một từ điển.
        values = dict(self)

        # data_config, decoder_config: Lấy cấu hình dữ liệu và cấu hình decoder từ từ điển values.
        data_config, decoder_config = values.get('data'), values.get('decoder')

        # using_text_embeddings: Kiểm tra xem có sử dụng nhúng văn bản hay không bằng cách kiểm tra thuộc tính cond_on_text_encodings trong mỗi cấu hình U-Net.
        if not exists(data_config) or not exists(decoder_config):
            # Then something else errored and we should just pass through
            return values

        using_text_embeddings = any([unet.cond_on_text_encodings for unet in decoder_config.unets])
        # using_clip: Kiểm tra xem có sử dụng mô hình CLIP hay không.  
        using_clip = exists(decoder_config.clip)
        # img_emb_url, text_emb_url: Lấy đường dẫn đến các tệp nhúng hình ảnh và văn bản.
        img_emb_url = data_config.img_embeddings_url
        text_emb_url = data_config.text_embeddings_url

        if using_text_embeddings:
            # Then we need some way to get the embeddings
            # Nếu sử dụng điều kiện hóa văn bản, phải cung cấp mô hình CLIP hoặc đường dẫn đến nhúng văn bản. 
            assert using_clip or exists(text_emb_url), 'If text conditioning, either clip or text_embeddings_url must be provided'

        if using_clip:
            if using_text_embeddings:
                assert not exists(text_emb_url) or not exists(img_emb_url), 'Loaded clip, but also provided text_embeddings_url and img_embeddings_url. This is redundant. Remove the clip model or the text embeddings'
            else:
                assert not exists(img_emb_url), 'Loaded clip, but also provided img_embeddings_url. This is redundant. Remove the clip model or the embeddings'

        if text_emb_url:
            assert using_text_embeddings 