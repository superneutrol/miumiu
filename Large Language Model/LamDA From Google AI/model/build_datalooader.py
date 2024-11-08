import copy 
from itertools import chain 
from datasets import load_dataset 
from torch.utils.data import DataLoader , DistributedSampler 
from torch.distributed import get_world_size
from config.config import CFG
from transformers import AutoTokenizer, default_data_collator

# Xây dựng phương thức để tải và xử lý sơ bộ dữ liệu 
# từ nguồn tham số được cấu hình 
def build_dataloader(args: CFG, tokenizer: AutoTokenizer):
    """
    build dataloaders for the model LamDA 
    """

    # Tải dữ liệu huấn luyện từ từ điển cấu hình hoặc một dang sách chứa 
    # các cấu hình sẽ được sử dụng cho mô hình lớp (CFG)
    load_train_data = load_dataset(args.train_dataset_name, split=args.choose_train_split)

    # Xóa các cột không được sử dụng từ tập dữ liệu train 
    load_train_data = load_train_data.remove_columns(args.remove_train_columns)

    # Tải xuống bộ dữ liệu xác thực 
    load_eval_data = load_dataset(args.eval_dataset_name, splt=args.choose_eval_split)

    # Xóa các cột không được sử dụng trong tệp xác thực 
    load_eval_data = load_eval_data.remove_columns(args.remove_eval_columns)

    # Shuffle the training input files. 
    # xáo chộn các data_train với  seed đảm bảo sau mỗi lần chạy thì thứ tự 
    # của dữ liệu được trộn sẽ luôn giống nhau 
    shuffled_train_files = load_train_data.shuffle(seed = args.seed)

    # Shuffle the validation input files.
    # tương tự như trên 
    shuffled_eval_files = load_eval_data.shuffle(seed = args.seed)

    """
    A sequence length of x is used for the model. Input examples are concatenated
    together and then split into sequences of exactly x tokens, so that there are 
    no padding tokens, but examples may be split in the middle.

    Tokenize function reference:
    https://github.com/hpcaitech/PaLM-colossalai/blob/main/data/wikitext.py
    """

    # Thiết lập phương thức tokenize để mã hóa các mẫu đầu vào
    # thành các chuỗi tokens x 
    def tokenize(examples):
        # lấy ra dộ dài cho phép của 1 chuỗi 
        seq_length = args.tokenizer_seq_length
        # lây sra các chuỗi string được định dạng text trong examples 
        examples = tokenizer(examples[args.select_input_string])
        #  Nối các chuỗi string lại với nhau dưới dạng 1 từ điển theo 1 khóa duy nhất 
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # lấy ra độ dài của dnah sách concatenated 
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # nếu như độ dài của mỗi examples trong danh sách mà lớn hơn độ dài cho phép 
        if total_length >= seq_length: 
            # Gán lại cho total length = với seq_length 
            total_length = (total_length // seq_length) * seq_length

        # khởi tạo một từ điển result để chứa đựng các kết quả theo một khóa keys 
        # với giá trị là một danh sách các chuỗi string có thứ tự được lấy mẫu 
        result ={
            # k là keys của danh sách các mẫu trong từ điển concat 
            k: [t[i: i + seq_length] for i in range(0, total_length, seq_length)]
            # t là danh sách chứa các chuỗi string đã được lấy mẫu và nhóm danh sách trong từ điển concatenated
            for k , t in concatenated_examples.items()
        }

        # thêm vào từ điển result 1 keys = labels và values = "inputs_ids" được sao chép 
        result["labels"] = copy.deepcopy(result["labels"])

        return result 
    

    """
    Map the tokenization function to the shuffled training files to create an 
    Iterable training dataset of batched input sequences of x tokens.
    Remove columns from the the shuffled training files so that you are left with 
    only the input_ids, attention_mask, and labels columns.
    """
    
    # áp dụng một hàm map cho việc xáo trộn dữ liệu train từ kết quả của hàm tokenize 
    # với batch_size và số lượng được chỉ định 
    tokenized_train_dataset = shuffled_train_files.map(tokenize, batched = True, remove_columns = [args.select_input_string])

    """
    Map the tokenization function to the shuffled validation files to create an 
    Iterable validation dataset of batched input sequences of x tokens.
    Remove columns from the the shuffled training files so that you are left with 
    only the input_ids, attention_mask, and labels columns.
    """
    
    # tương tự như trên và cũng chỉ dữ lại các cột input_ids, attention_mask, và labels 
    tokenized_eval_dataset = shuffled_eval_files.map(tokenize, batched = True, remove_columns = [args.select_input_string])

    # chuyển đổi định dạng của dữ liệu train đã được mã hóa sang Pytorch Tensor 
    train_with_torch = tokenized_train_dataset.set_format(type="torch")

    # Convert the format of the tokenized validation dataset to PyTorch Tensors
    eval_with_torch = tokenized_eval_dataset.set_format(type = "torch")

    # Train dataset used for sampling.
    # thực hiện lấy mẫu accs tensor data Train và phân phối nó cho thiết bị trong môi trường 
    sample_train_dataset = DistributedSampler(train_with_torch, shuffle = True) if get_world_size() > 1 else None

    # Validation dataset used for sampling.
    # tương tự như trên ta áp dụng lấy mẫu phân phối cho dữ liệu xác thực 
    sample_eval_dataset = DistributedSampler(eval_with_torch, shuffle = False) if get_world_size() > 1 else None

    # Create the train dataloader. If the length of a tokenized input sequence is less than 2048 drop it.
    train_dataloader = DataLoader(tokenized_train_dataset, shuffle = True, sampler = sample_train_dataset, drop_last = True, collate_fn = default_data_collator, batch_size = args.batch_size)

    # Create the validation dataloader. If the length of a tokenized input sequence is less than 2048 drop it.
    eval_dataloader = DataLoader(tokenized_eval_dataset, sampler = sample_eval_dataset, drop_last = True, collate_fn = default_data_collator, batch_size = args.batch_size)

    # Return the training and validation dataloaders to be used in the model
    print('Done building dataloaders')
    return train_dataloader, eval_dataloader


# kiểm tra xen python có đang chạy như là một chương trình chính hay không 
if __name__ == "__main__":
    # Lấy ra các cấu hình ho dataloader 
    data_loader_args = CFG()

    #Lấy các cấu hình mã hóa tham số 
    tokenizer_args = "gpt2" # sử dụng kiểu mã hóa Byte Pair Encoding của GPT2

    # Tải mã thông báo tokens đã được đào tạo trước 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_args)

    # Build data_loader 
    train_loader, eval_loader = build_dataloader(args= data_loader_args, tokenizer=tokenizer)


    # Dòng này in ra input_ids của batch đầu tiên từ data loader huấn luyện.
    print(next(iter(train_loader))['input_ids'])
    # Tương tự, dòng này in ra kích thước (chiều) của tensor input_ids từ batch đầu tiên.
    print(next(iter(train_loader))['input_ids'].shape)
