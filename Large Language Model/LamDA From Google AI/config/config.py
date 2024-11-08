from __future__ import annotations 
from typing import Optional, ClassVar 
from dataclasses import dataclass, field 

# xây dựng một đối tượng dataclass từ thư viện dataclass
# để đingj nghĩa một lớp cấu hình 
@dataclass 
class CFG: 
    """
    Configuration for zero 

    """
    # Đây là một biến câu hình để xác định liệu có sử dụng zeRO hay không 
    # ZEro LÀ một kỹ thuật tối ưu hóa bộ nhớ cho việc huấn luyện mô hình 
    # Giá trị mặc định là False, tức là không sử dụng ZeRO nếu không được chỉ định.
    use_zero: bool = field(
        default=False,
        metadata={'help': 'whether to use zero'}
    )

    """
    Configuration for optimizer
    """
    # cấu hình một tham số lr 
    lr: float = field(
        default = 0.0001,
        metadata = {'help': 'learning rate'}
    )

    """
    Configuration class for LaMDA model.
    """
    # num_tokens 
    num_tokens: int = field(
        default = 50257,
        metadata = {'help': 'number of tokens'}
    )
    # hidden_dim 
    dim: int = field(
        default = 512,
        metadata = {'help': 'dimension of the embedding'}
    )
    # depth layer 
    depth: int = field(
        default = 6,
        metadata = {'help': 'depth of the transformer'}
    )
    # number headers for multihead self-attentention 
    heads: int = field(
        default = 4,
        metadata = {'help': 'number of heads in the transformer'}
    )
    # hidden_dim for per header 
    dim_head: int = field(
        default = 64,
        metadata = {'help': 'dimension of the head'}
    )

    """
    Configuration for data loader.
    """

    use_huggingface: bool = field(
        default = True,
        metadata = {'help': 'Whether to use huggingface datasets'}
    )

    train_dataset_name: Optional[str] = field(
        default="the_pile", 
        metadata={"help": "Path to Hugging Face training dataset."}
    )

    eval_dataset_name: Optional[str] = field(
        default="the_pile", 
        metadata={"help": "Path to Hugging Face validation dataset."}
    )

    choose_train_split: Optional[str] = field(
        default="train", 
        metadata={"help": "Choose Hugging Face training dataset split."}
    )

    choose_eval_split: Optional[str] = field(
        default="train", 
        metadata={"help": "Choose Hugging Face validation dataset split."}
    )

    remove_train_columns: ClassVar[list[str]] = field(
        default = ['meta'], 
        metadata={"help": "Train dataset columns to remove."}
    )

    remove_eval_columns: ClassVar[list[str]] = field(
        default = ['meta'], 
        metadata={"help": "Validation dataset columns to remove."}
    )

    seed: Optional[int] = field(
        default=42, 
        metadata={"help": "Random seed used for reproducibility."}
    )

    tokenizer_name: Optional[str] = field(
        default="gpt2",
        metadata={"help": "Tokenizer name."}
    )
    # Tokennizer with per seqence leangth inputs 
    tokenizer_seq_length: Optional[int] = field(
        default=512, 
        metadata={"help": "Sequence lengths used for tokenizing examples."}
    )

    select_input_string: Optional[str] = field(
        default="text", 
        metadata={"help": "Select the key to used as the input string column."}
    )
    # batch_size 
    batch_size: Optional[int] = field(
        default=16, 
        metadata={"help": "Batch size for training and validation."}
    )
    # 
    save_to_path: Optional[str] = field(
        default="''", 
        metadata={"help": "Save the dataset to local disk."}
    )

    """
    Configuration for Weights and Biases
    """

    use_wandb: bool = field(
        default = False,
        metadata = {'help': 'Whether to use Weights and Biases for logging'}
    )
    # Đây là một thuộc tính cho tên dự án.
    # Kiểu dữ liệu là Optional[str], nghĩa là có thể là một chuỗi hoặc None.
    # Giá trị mặc định là "LaMDA pre-training".
    # Metadata cung cấp thông tin trợ giúp về thuộc tính này.
    project_name: Optional[str] = field(
        default="LaMDA pre-training",
        metadata={'help': 'Name of the project'}
    )