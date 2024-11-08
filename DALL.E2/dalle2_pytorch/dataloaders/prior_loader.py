from math import ceil
from typing import Iterator 
from clip import tokenize 
from embedding_reader import EmbeddingReader
from torch import from_numpy
from torch.utils.data import IterableDataset, DataLoader


# Thiết lập lớp phương thức PriorEmbeddingDataset được sử dụng để tạo ra một 
# tập dữ liệu có thể lặp cho phép lấy mẫu chúng một cách dễ dàng 
# lớp này kế thừa từ lớp IterableDataset trong pytorch sử dụng như một bao bọc 
# cho embeddingreader giúp đơn giản hóa logic cần thiết 
class PriorEmbeddingDataset(IterableDataset): 
    """
    PriorEmbeddingDataset is a wrapper of EmbeddingReader.

    It enables one to simplify the logic necessary to yield samples from
    the different EmbeddingReader configurations available.
    """
    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính tham số
    def __init__(
        self,
        text_conditioned: bool,
        batch_size: int,
        start: int,
        stop: int,
        image_reader,
        text_reader: EmbeddingReader = None,
    ) -> None: # tham số none chỉ định hàm không trả về 
        # cho phép lớp này được kế thừa lại 
        super(PriorEmbeddingDataset).__init__()
        # định nghĩa thuộc tính text_condition = bool 
        # thuộc tính này cho biết mô hình sẽ học từ duy nhất các ảnh hay là cả ảnh và dữ liệu 
        # nếu = True là chỉ tập trung học với các ảnh còn False là học từ các embedding văn bản và ảnh 
        self.text_conditioned = text_conditioned

        # nếu như thuộc tính này không tồn tại 
        if not self.text_conditioned: 
            # tức nó = False thì khởi tạo thuộc tính text reader 
            self.text_reader = text_reader
        
        # định nghĩa thuộc tính image_reader 
        self.image_reader = image_reader
        # tiếp theo là chỉ số bắt đầu vào chỉ số kết thúc 
        self.start = start
        self.stop = stop
        # Và cuối cùng là kích thước lô đầu vào 
        self.batch_size = batch_size

    # Thiết lập phương thức len sẽ trả về số lượng mẫu trong tập dữ liệu 
    # được quy định bởi vị trí được lấy và vị trí mẫu kết thúc 
    def __len__(self):
        return self.stop - self.start 
    
    # Thiết lập phương thức iter như là một phương thức lặp 
    # sau đó trả về đối tượng cho phép nó được sử dụng như một iterator (lặp)
    def __iter__(self):
        # khởi tạo một bộ nạp dữ liệu loader 
        # dựa trên các tham số được thiết đặt 
        loader_args = dict(
            batch_size=self.batch_size,
            start=self.start,
            end=self.stop,
            show_progress=False,
        )
        # if the data requested is text conditioned, only load images
        if self.text_conditioned:
            # sử dụng toán tử ** để nén các đối số được chuyền vào loader_args 
            # với toán tử này nó sẽ tự động tìm kiếm các đối số tương ứng 
            self.loader = self.image_reader(**loader_args)
        # nếu không thì 
        # otherwise, include text embeddings and bypass metadata
        else:
            self.loader = zip(
                self.image_reader(**loader_args), self.text_reader(**loader_args)
            )

        # return the data loader in its formatted state
        return self
    

    # Thiết lập phương thức __next__ để lấy mẫu tiếp theo từ tập dữ liệu 
    def __next__(self):
        try:
            # nó gọi đến phương thức get_sample để tiến hành lấy mẫu 
            return self.get_sample()

        # nếu như không còn mẫu nào để lấy nén ra một ngoại lệ StopIteration  
        except StopIteration: # điều này cho phép lớp này có thể hoạt động như một iterator 
            # trong python 
            raise StopIteration

    # thiết lập phương thức str nó sẽ trả về một chuỗi biểu diễn của dối tượng 
    # PriorEmbeddingDataset
    def __str__(self):
        # Và Python sẽ gọi phương thức này để lấy chuỗi biểu diễn.
        return f"<PriorEmbeddingDataset: start: {self.start}, stop: {self.stop}, len: {self.__len__()}>"

    def set_start(self, start):
        """
        Adjust the starting point within the reader, useful for resuming an epoch
        """
        # Phương thức này cho phép điều chỉnh điểm bắt đầu trong bộ đọc, 
        # điều này hữu ích khi muốn tiếp tục một epoch. 
        # Nó cập nhật giá trị của thuộc tính start của đối tượng
        self.start = start

    # : Phương thức này trả về giá trị hiện tại của thuộc tính start của đối tượng.
    def get_start(self):
        return self.start#


    # Thiết lập phương thức get_sample sử dụng cho việc tiền xử lý dữ liệu
    def get_sample(self):
        """
            pre-process data from either into a common format 
        """ 
        # kiểm tra xem text_conditioned = True hay không 
        if self.text_conditioned: 
            # thì ta duyệt qua mẫu tiếp theo và tải nó bao gồm image và captionn for image 
            image_embedding, caption = next(self.loader)
            
            # Thực hiện lấy các image_embedding dưới dạng mảng numpy và chuyển nó sang Tensor
            image_embedding = from_numpy(image_embedding)
            # token hóa các caption theo hình ảnh 
            tokenized_caption = tokenize(caption["caption"].to_list(), truncate=True)

            # Trả về hình ảnh nhúng và danh sách nhúng các token của caption 
            return image_embedding, tokenized_caption
        
        # trường hợp text_conditioned = False mô hình sẽ tập trung học từ các embedding văn bản và ảnh 
        else: 
            # Thực hiện việc lấy mẫu tiếp theo từ loader 
            # hàm này sẽ trả về 1 tuple mỗi phần tử được gán cho image_embedding và text_embedding 
            (image_embedding, _), (text_embedding, _) = next(self.loader) 

            # sau đó chuyển đổi các embedding ảnh và văn bản từ numpy array sang tensor 
            image_embedding = from_numpy(image_embedding)
            text_embedding = from_numpy(text_embedding)

            # kết quả trả về một tuple chứa embedding ảnh và embedding văn bản 
            return image_embedding, text_embedding
        


# Helper functions 

# Thiết lập phương thức distribute_to_rank có chức năng thực hiện 
# việc phân bổ dữ liệu đến mỗi cấp bậc theo quy mô của môi trường phân tán 
# tham số world_size có thể hiểu là số lượng thiết bị trong môi trường, còn rank sẽ tương ứng với 
# chỉ số thiết bị hoặc thứ tự thiết bị trong word_size 
def distribute_to_rank(start, stop, rank, world_size):
    """
    Distribute data to each rank given the world size.

    Return:
        - New start and stop points for this rank.
    """

    # Tính toán số lượng mẫu được tính từ chỉ số được lấy đầu tiên đến chỉ số cuối cùng được chỉ định 
    num_samples = int(stop - start)

    # Tính số lượng mẫu dữ liệu mà mỗi rank sẽ nhận được. Và nó được làm tròn lên để đảm bảo rằng 
    # mỗi ranh có thể nhận ít nhất một mẫu 
    per_rank = int(ceil((num_samples) / float(world_size)))

    # kiểm tra và đảm bảo rằng mỗi rank nhận được ít nhất một mẫu 
    assert (per_rank > 0), f"Number of samples per rank must be larger than 0, (found: {per_rank})"

    # Tính điểm bắt đầu cho rank hiện tại bằng với điểm bắt đầu tổng thể + thứ tự của ranh 
    # và nhân với số lượng mẫu của rank hiện tại 
    # VD. start = 0 per_rank = 10 và rank là 2 thì kết quả của điểm bắt đầu cho rank 2 = 20 (idx)
    rank_start = start + rank * per_rank 

    # Tính điểm kết thúc cho rank hiện tại đảm bảo nó không vượt quá stop 
    rank_stop = min (rank_start + per_rank, stop)

    # Và tính độ dài mới cho rank 
    new_length = rank_stop - rank_start 

    # kiểm tra và đảm bảo rằng độ dài mới phải lớn hơn không 
    assert (
        new_length > 0
    ), "Calculated start and stop points result in a length of zero for this rank."

    # Trả về điểm bắt đầu và kết thúc của rank hiện tại 
    return rank_start, rank_stop
    


# Thiết lập phương thức get_reader để nhúng các embedding văn bản và image embedding 
# kết quả trả về sẽ phụ thuộc vào text_conditioned 
def get_reader( text_conditioned: bool, img_url: str, meta_url: str = None, txt_url: str = None
):
    """
    Create an EmbeddingReader object from the specified URLs

    get_reader() will always expect a url to image embeddings.

    If text-conditioned, it will also expect a meta_url for the captions.
    Otherwise, it will need txt_url for the matching text embeddings.

    Returns an image_reader object if text-conditioned.
    Otherwise it returns both an image_reader and a text_reader
    """
    # đảm bảo rằng đường dẫn url của hình ảnh có tồn tại 
    assert img_url is not None, "Must supply a image url" # phải cung cấp một url hình ảnh 

    # Kiểm tra xem text_conditioned = True hay không 
    if text_conditioned: 
        # nếu nó có tồn tại thì chúng ta phải đảm bảo rằng một đường dẫn meta phải tồn tại 
        assert meta_url is not None, "Must supply meta url if text-conditioned"

        # với text_conditoned cho ta biết ở đây chúng ta đang tập chung vào học 
        # các biểu diễn nhúng hình ảnh và giảm thiểu sự tập chung vào các caption của nó 
        image_reader = EmbeddingReader(
            embeddings_folder=img_url,
            file_format="parquet_npy",
            # will assume the caption column exists and is the only one requested
            meta_columns=["caption"],
            metadata_folder=meta_url,
        )

        # Trả về danh sách nhúng của hình ảnh 
        return image_reader
    
    # Nếu không ta phải thực hiện cả nhúng văn bản và hình ảnh sau đó trả về 2 trình 
    # đọc tương ứng với chúng 
    assert (
        # đảm bảo rằng đường dẫn txt_url có tồn tại 
        txt_url is not None
    ), "Must supply text embedding url if not text-conditioning"
    
    # thực hiện nhúng văn bản và hình ảnh sau đó gán chúng cho trình đọc tương ứng 
    image_reader = EmbeddingReader(img_url, file_format="npy")
    text_reader = EmbeddingReader(txt_url, file_format="npy")

    # cuối cùng trả về 2 trình đọc 
    return image_reader, text_reader


# Thiết lập phương thức make_splits để tách một đối tượng trình đọc nhúng nếu cần 
# kết quả cuối cùng trả về 1tuple và mang lại bộ dữ liệu (img, txt). 
def make_splits(
    text_conditioned: bool,
    batch_size: int,
    num_data_points: int,
    train_split: float,
    eval_split: float,
    image_reader: EmbeddingReader,
    text_reader: EmbeddingReader = None,
    start=0,
    rank=0,
    world_size=1,
):
    """
    Split an embedding reader object as needed.

    NOTE: make_splits() will infer the test set size from your train and eval.

    Input:
        - text_conditioned: whether to prepare text-conditioned training data
        - batch_size: the batch size for a single gpu
        - num_data_points: the total number of data points you wish to train on
        - train_split: the percentage of data you wish to train on
        - eval_split: the percentage of data you wish to validate on
        - image_reader: the image_reader you wish to split
        - text_reader: the text_reader you want to split (if !text_conditioned)
        - start: the starting point within your dataset
        - rank: the rank of your worker
        - world_size: the total world size of your distributed training run

    Returns:
        - PyTorch Dataloaders that yield tuples of (img, txt) data.
    """
    # ĐẢM BẢO RẰNG CHỈ SÓ BẮT ĐẦU có vượt quá số lượng dữ liệu trong image_reader hay không 
    assert start < image_reader.count, "start position cannot exceed reader count."

    # xác minh rằng num_data_point không vượt số lượng điểm tối đa từ 
    # start đến cuối image_reader hay không 
    if num_data_points > (image_reader.count - start):
        print(
            "Specified count is larger than what's available...defaulting to reader's count."
        )
        # sau đó gán lại cho num_data_point bằng với số lượng dữ liệu trong image_reader 
        num_data_points = image_reader.count 
    
    # Thực hiện tính toán và phân chia dữ liệu cho mô hình 
    # 1: Tính kích thước huấn luyện bằng cách nhân tỷ lệ huấn luyện với tổng số 
    # dữ liệu 
    train_set_size = int(train_split * num_data_points)
    # 2: tương tự tính kích thước của tập đánh giá 
    eval_set_size = int(eval_split * num_data_points)
    # 3: tính toán điểm bắt đầu của tập đánh giá bắt đầu từ điển cuối cùng + 1 của 
    # tập huấn luyện
    eval_start = train_set_size
    # 4: và điểm cuối cùng sẽ được tính bằng điểm bắt đầu + với kích thước bộ dữ liệu đnahs giá 
    eval_stop = int(eval_start + eval_set_size)

    # đảm bảo rằng tỷ lệ tách của dữ liệu train và evaluation không vượt quá 100 % 
    assert (
        train_split + eval_split
    ) < 1.0, "Specified train and eval split is too large to infer a test split."

    # distribute to rank
    # sử dụng hàm distribute_to_rank để phân phối dữ liệu cho các rank trong môi trường 
    # thiết bị world size
    rank_train_start, rank_train_stop = distribute_to_rank(
        start, train_set_size, rank, world_size
    )
    # Và tương tự ta áp dụng việc này cho bộ dữ liệu evaluation 
    rank_eval_start, rank_eval_stop = distribute_to_rank(
        train_set_size, eval_stop, rank, world_size
    )
    # cuối cùng là dữ liệu thử nghiệm 
    rank_test_start, rank_test_stop = distribute_to_rank(
        eval_stop, num_data_points, rank, world_size
    )

    # Thực hiện việc gom các dữ liệu được tính toán từ start đến stop 
    # lại dưới dạng một từ điển từ điển này sẽ chứa thông tin về điểm bắt đầu và kết thúc 
    # của mỗi bộ dữ liệu 
    # 1: Xác định với train_data
    train_split_args = dict (
        start=rank_train_start, stop=rank_train_stop, batch_size=batch_size
    )
    # 2: Xác định với eval_data
    eval_split_args = dict(
        start=rank_eval_start, stop=rank_eval_stop, batch_size=batch_size
    )
    # 3: Xác định với test_data 
    test_split_args = dict(
        start=rank_test_start, stop=rank_test_stop, batch_size=batch_size
    )

    # Nếu như text_conditioned = True tức là dữ liệu được điều kiện hóa bởi văn bản 
    # thì sẽ thêm các tham số là image_reader và text_conditioned vào từ điển reader_args 
    if text_conditioned:
        # add the text-conditioned args to a unified dict
        reader_args = dict(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
        )
        # sau đó thêm từ điển reader_args chứa thôngb tin của text_conditioned và image_reader 
        # vào trong các từ điển chứa thông tin đểtaoj ra các từ điển hoàn chính chứa tất cả các 
        # thông tin cần thiết cho việc xây dựng bộ dữ liệu 
        train_split_args = dict(**reader_args, **train_split_args)
        eval_split_args = dict(**reader_args, **eval_split_args)
        test_split_args = dict(**reader_args, **test_split_args)

        # Cuối cùng thực hiện nhúng các bộ dữ liệu cho mô hình 
        train = PriorEmbeddingDataset(**train_split_args)
        val = PriorEmbeddingDataset(**eval_split_args)
        test = PriorEmbeddingDataset(**test_split_args)

    # trường hợp còn lại với văn abnr không được điều kiện hóa 
    else: # trường hợp này sử dụng cả văn bản và hình ảnh để huấn luyện mô hình 
        # Thêm các đối số không có điều kiên vào một lệnh thống nhất q
        # add the non-conditioned args to a unified dict
        reader_args = dict(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
            text_reader=text_reader,
        )

        # Sau đó thêm thôm tin của từ điển reader_args và các từ điển điến chứa thông tin khác 
        # vào một từ điển lưu trữ đầy đủ 
        train_split_args = dict(**reader_args, **train_split_args)
        eval_split_args = dict(**reader_args, **eval_split_args)
        test_split_args = dict(**reader_args, **test_split_args)

        # cuối cùng thực hiện nhúng Prior cho các tập dữ liệu 
        train = PriorEmbeddingDataset(**train_split_args)
        val = PriorEmbeddingDataset(**eval_split_args)
        test = PriorEmbeddingDataset(**test_split_args)


    # Cuối cùng xây dựng bộ dữ liệu hoàn chỉnh
    train_loader = DataLoader(train, batch_size=None)
    eval_loader = DataLoader(val, batch_size=None)
    test_loader = DataLoader(test, batch_size=None)

    # và sau đó là trả về các bộ dữ liệu này 
    return train_loader, eval_loader, test_loader

