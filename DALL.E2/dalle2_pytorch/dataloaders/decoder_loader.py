import os 
import webdataset as wds 
import torch 
from torch.utils.data import DataLoader 
import numpy as np 
import fsspec 
import shutil 

# thiết lập phương thức gẻ_shard để chia các phân đoạn có cấu 
# trúc nhất quán từ file name 
def get_shard(filename):
    """
        Filename with shards in them have a consistent structure that we can take advantage 
        of Standard structure: path/to/file/prefix_string_00001.ext
    """
    # thực hiện một khối ngoại lệ 
    try: 
        # thực hiện tách các phần văn bản của danh sách file name theo _ 
        # và ở lần tách tiếp theo ta tách theo giấu. và giữ lại phần tử đầu tiên 
        # kết quả sẽ trả về là một chuỗi 00001 
        return filename.split('_')[-1].split(".")[0]
    except  ValueError: # nếu quá trình trên gặp lỗi nếu tên file không chứa _ hoặc . 
        # hàm sẽ ném ra một lỗi RuntimeError với thông báo 
        raise RuntimeError(f"Could not find shard for filename {filename}")
    

# Thiết lập phương thức get_example_file nhận một file từ hệ thống 
# và mọt file mở rộng sau đó trả về file example 
def get_example_file(fs, path, file_format):
    """
    Give a file system and a file extention, return the example file 

    """
    # sử dụng hàm join để nối đường dẫn mở rộng với một định dạng file và * ký tự đại diện 
    # nghĩa là nó sẽ khớp với một chuỗi ký tự bất kỳ nào. VD nếu file_format là txt thì chuỗi này 
    # sẽ khớp với tất cả các file có định dạng txt 
    # hàm glob() sẽ tìm kiếm toàn bộ các file trong file system mà đường dẫn của chúng khớp 
    # với đường dẫn đã đạo bởi join(). Cuối cùng tham số [0] để trả 
    # về phần tử file đầu tiên khớp với định dạng file format trong path 
    return fs.glob(os.path.join(path, f"*.{file_format}"))[0]


# khởi tạo phương thức embedding_inserter nghĩa là bộ trèn embedding
def embedding_inserter(samples, embeddings_url, index_width, sample_key='npy', handler=wds.handlers.reraise_exception):
    """Given a datum of {"__key__": str, "__url__": str, ...} adds the cooresponding embedding and yields"""
    # khởi tạo một đường dẫn url trước 
    previous_tar_url = None 
    # và một đường dẫn url chỉ định hiện tại 
    current_embeddings = None 
    # Nhận một tham chiếu đến hệ thống tệp trừu tượng nơi lưu trữ các phần nhúng 
    # sử dụng hàm fsspec.core.url_to_fs(embedding_url) để lấy ra một tham chiếu đến hệ thống 
    # file nơi các embedding được lưu trữ
    embeddings_fs, embeddings_path = fsspec.core.url_to_fs(embeddings_url)
    # thực hiện gọi hàm get_example_file được gọi để lấy ra file embedding mẫu 
    # kết quả trả về đường dẫn đầy đủ của file này 
    example_embedding_file = get_example_file(embeddings_fs, embeddings_path, "npy")
    # gọi hàm get_shards để tách ra phần shard từ tên file embedding mẫu 
    example_embedding_shard = get_shard(example_embedding_file)
    # lấy ra số lượng ký tự trong độ dài shard 
    emb_shard_width = len(example_embedding_shard)
    # lấy ra đường dẫn embedding cơ sở không bao gồm phần shard 
    embedding_file_basename = '_'.join(example_embedding_file.split("_")[:-1])+ "_"

    
    # Thiết lập phương thức load_corresponding_embeds để tải các file npy embedding 
    # cho tar là tập dữ liệu web đã cho 
    def load_corresponding_embeds(tar_url):
        """Finds and reads the npy files that contains embeddings for the given webdataset tar"""
        # lấy ra phần shard từ tar_url sau đó chuyển đổi nso thành số nguyên 
        # đầu tiên tách file url thành các phần theo / và lấy phần cuối cùng 
        # phần shard này sau đó được tách bỏ đuôi file và lấy tên file 
        shard = int(tar_url.split("/")[-1].split(".")[0]) # sau đó chuyển tên file thành số ký tự int 
        # tạo một chuỗi đường dẫn embedding_url 
        # bằng cách nối 2 chuỗi file base embedding không có shard và một chuỗi 
        # là số lượng shard.zerosfill(emb_shard_width) thực hiệm đệm một chuỗi shard theo số lượng 
        # ký tự tiêu chuẩn được lấy từ file_tar các giá trị đệm = 0 
        # nếu shard = 5 mà emb_shard_with của file embedding = 3 ['123'] thì sẽ có dạng '00123'
        embedding_url = embedding_file_basename + str(shard).zfill(emb_shard_width) + '.npy'
        # đọc dữ liệu của file embedding_url từ file hệ thống và gán nó cho biến f
        with embeddings_fs.open(embedding_url) as f:
            # kết quả là một mảng numpy chứa dữ liệu của embedding 
            data = np.load(f)
        # sau đó chuyển đổi các mảng numpy embedding thành các tensor embedding 
        return torch.from_numpy(data)


    # lặp qua các mẫu trong sample mỗi sample là một từ điển chứa các thông tin 
    # như URL của tệp tin ("__url__") và khóa ("__key__").
    for sample in samples: 
        # xây dựng một khối ngoại lệ 
        try: 
            # gán cho url tar = khóa url trong từ điển sample 
            # và key = khóa key trong từ điển 
            tar_url = sample["__url__"]
            key = sample["__key__"]
            #nếu url của tar khác với url của tệp trước đó 
            if tar_url != previous_tar_url: 
                # hàm sẽ tải các embedding mới băng cách gọi hàm load_corresponding_embed 
                # và cập nhật url sau đó thành url hiện tại 
                previous_tar_url = tar_url
                current_embeddings = load_corresponding_embeds(tar_url)

            # tính toán chỉ số embedding  = giá trị nguyên của từ điển key 
            # theo số lượng index_width cuối cùng 
            embedding_index = int(key[-index_width:])
            # sau đó lấy ra một danh sách các embedding hiện tại theo số lượng phần 
            # tử đươc cắt theo embedding_index 
            embedding = current_embeddings[embedding_index]

            # kiểm tra xem embedding có phải là một tensor không chứa giá trị khác 0 
            # nếu các giá trị = 0 hàm sẽ ném ra một lỗi 
            if torch.count_nonzero(embedding) == 0:
                raise RuntimeError(f"Webdataset had a sample, but no embedding was found. ImgShard: {key[:-index_width]} - Index: {key[-index_width:]}")
            
            # nếu không embedding sẽ được thêm vào từ điển sample với khóa là sample_key 
            sample[sample_key] = embedding
            # và sample sẽ không được trả về thông qua từ kháo yeild 
            yield sample
        except Exception as exn: # từ việc triển khai wds 
            # Nếu có bất kỳ lỗi nào xảy ra trong quá trình thực hiện các bước trên, hàm handler sẽ được gọi với lỗi đó làm tham số. 
            if handler(exn):
                # nếu handler = True vòng lặp sẽ tiếp tục lặp với mấu tiếp theo 
                # nếu không sẽ kết thúc 
                continue 
            else: 
                break 

# cuối cùng inser_embedding được khởi tạo như một bộc lọc pipeline sử dụng hàm 
# embedding_inserter. Bộ lọc này có thể được sử dụng để thêm embedding vào các mẫu trong một pipeline xử lý dữ liệu
insert_embedding = wds.filters.pipelinefilter(embedding_inserter)


# Thiết lập phương thức unassociated_shard_skipper được sử dụng để tìm và trả về 
# các tarfile mà có embedding tương ứng với nó 
# Nếu không có file embedding tương ứng nó sẽ bỏ qua 
# hàm này đảm bảo chỉ có các tarfile có embedding tương ứng mới được xử lý, giúp tăng hiệu suất và 
# chất lượng của quá trình xử lý 
def unassociated_shard_skipper(tarfiles, embeddings_url, handler=wds.handlers.resaise_exception):
    """Finds if the is a corresponding embedding for the tarfile at {url: [URL]}"""
    # gọi hàm fsspec.core.url_to_fs để lấy ra một tham chiếu đến hệ thống file nởi các embedding 
    # được lưu trữ  # kết quả là 1 tuple gồm file hệ thống và đường dẫn trong hệ thống đến các file embedding 
    embeddings_fs, embedding_path = fsspec.core.url_to_fs(embeddings_url)
    # lấy ra danh sách tất cả các file embedding trong embedding_path 
    embedding_files = embeddings_fs.ls(embedding_path)
    # Hàm get_embedding_shard được định nghĩa như một hàm lambda để lấy ra phần shard từ tên file 
    # embedding. Phần shard này sau đó sẽ được chuyển đổi thành số nguyên 
    get_embedding_shard = lambda embedding_file: int(embedding_file.split("_")[-1].split(".")[0])
    # sau đó lưu chữ các phần shard của các file embedding vào một dnah sách set 
    embedding_shards = set([get_embedding_shard(filename) for filename in embedding_files])  # Sets have O(1) check for member

    # hàm get_tar_shard được định nghĩa như một hàm lambda để lấy ra phần shard từ url của một tập tin  tar 
    get_tar_shard = lambda tar_file: int(tar_file.split("/")[-1].split(".")[0])
    # lặp qua tất acr các targiles trong tarfiles. 
    for tarfile in tarfiles:
        try:
            # Đối với mỗi tarfile hàm lấy ra phần shard từ URL của tarfile 
            webdataset_shard = get_tar_shard(tarfile["url"])
            #và kiểm tra xem phần  shard này có tồn tại trong embedding_shard hay không 
            if webdataset_shard in embedding_shards:
                # Nếu có tarfile sẽ được trả về thông qua từ khóa yield 
                yield tarfile
        
        # Nếu có bất kỳ lỗi nào xảy ra trong quá trình thực hiện các bước trên
        except Exception as exn:  # From wds implementation
            # hàm handler sẽ được gọi với lỗi đó làm tham số
            if handler(exn):
                # nếu handler trả về True vòng lặp sẽ tiếp tục 
                continue
            # còn lại sẽ dừng lập tức vòng lặp 
            else:
                break

# cuối cùng phương thức này được khởi tạo như một bộ lọc pipeline
skip_unassociated_shards = wds.filters.pipelinefilter(unassociated_shard_skipper)


# Thiết lập phương thức join_embedding để nhóm các vector nhúng biểu diễn 
# của img_emb và text_embed thành một từ điển nhúng biểu diễn "emb": { "text": text_emb, "img": img_emb }
def join_embeddings(samples, handler=wds.handlers.reraise_exception):
    """
    Takes the img_emb and text_emb keys and turns them into one key "emb": { "text": text_emb, "img": img_emb }
    either or both of text_emb and img_emb may not be in the sample so we only add the ones that exist
    """
    # duyệt qua danh sách các từ điển sample 
    for sample in samples:
        try:
            # thêm 1 khóa embed vào từ điển với values = none
            sample['emb'] = {}
            # kiểm tra xem trong từ điển này có chứa text_embed values hay không 
            if 'text_emb' in sample:
                # nếu có thêm từ điển embe key = text và values = text_embed là dữ liệu nhúng của văn bản 
                sample['emb']['text'] = sample['text_emb']
            # nếu như từ điển này có chứa image_embed values 
            if 'img_emb' in sample:
                # nếu có chứa thêm vào từ điển emb key = img values = image_embed
                sample['emb']['img'] = sample['img_emb']
            
            # và từ điển sample sẽ được trả thông qua từ khóa yield 
            yield sample
        # Nếu có một lỗi nào xảy ra trong quá trình thực thi các bước trên 
        except Exception as exn:  # From wds implementation
            # hàm handler sẽ được gọi nếu nó = True 
            if handler(exn):
                # tức không lỗi thì tiếp tục duyệt qua dnah sách sample
                continue
            # nếu không ta kết thúc 
            else:
                break



# Thiết lập phương thức verify_keys được sử dụng để đảm bảo rằng cả hình ảnh và embedding đều có 
# mặt trong từ điển mẫu. 
def verify_keys(samples, required_keys, handler= wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    # duyệt qua dnah sách các từ điển sample 
    for sample in samples:
        try:
            # đối với mỗi từ điển sample kiểm tra tất cả các keys trong danh sách 
            # required_keys có tồn tại trong sample hay không
            for key in required_keys:
                # đảm baỏ rằng sample có tồn tại 1 keys trong danh sách required_keys 
                assert key in sample, f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}"
                # Nếu có một keys không tồn tại hàm sẽ ném ra một lỗi AssertionError với thông báo lỗi 
                # tương ứng 
            #
            # nếu không có lỗi xảy ra sample sẽ được trả về thông qua từ khóa yield 
            yield sample

        # Nếu có bất kỳ lỗi nào xảy ra trong bất kỳ các bước trên
        except Exception as exn:  # From wds implementation
            # hàm handler sẽ được gọi với lỗi đó là tham số 
            if handler(exn):
                continue
            # nếu không vòng lặp sẽ kết thúc 
            else:
                break

# cuối cùng key_verify được khởi tạo như một bộ lọc pipeline sử dụng hàm verify_keys 
# bộ lọc này có thể được sử dụng để  kiểm tra xem các mẫu dữ liệu có chứa tất cả các khóa yêu cầu hay không.
key_verifier = wds.filters.pipelinefilter(verify_keys)


# Thiết lập lớp phương thức Nhúng các dữ liệu hình ảnh trả về các cặp nhúng hình ảnh 
# đọc các phần nhúng dưới dạng tệp npy từ tệp dữ liệu web nếu chúng tồn tại
class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """
    # thiết lập phương thức khởi tạo và đinhj nghĩa các thuộc tính tham số 
    def __init__(
            self,
            urls,
            img_embedding_folder_url=None,
            text_embedding_folder_url=None,
            index_width=None,
            img_preproc=None,
            extra_keys=[],
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True
    ):
        """
        Modeled directly off of the WebDataset constructor

        :param urls: A url pointing to the tar files of the webdataset formatted as /path/to/webdataset/{0000..9999}.tar
        :param embedding_folder_url: Required if webdataset does not contain embeddings. A url pointing to the npy files of the embeddings. Should have the same number of shards as the webdataset.
            Webdataset image keys should align with the index of the embedding. This means missing image indices must have a corresponding embedding of all zeros.
        :param index_width: The number of digits in the index. This is used to align the embedding index with the image index.
            For example, if a file in the webdataset shard 3 is named 0003039.jpg, we know the shard is 4 digits and the last 3 digits are the index_width.
        :param img_preproc: This function is run on the img before it is batched and returned. Useful for data augmentation or converting to torch tensor.
        :param handler: A webdataset handler.
        :param resample: If true, resample webdataset shards with replacement. You need to set your own epoch size if this is true since it will resample infinitely.
        :param shuffle_shards: If true, shuffle the shards before resampling. This cannot be true if resample is true.


        """
        super().__init__()
        keys = ["jpg", "emb"] + extra_keys 
        # if img_embedding_folder_url is not None:
        #     keys.append("img_emb")
        # if text_embedding_folder_url is not None:
        #     keys.append("text_emb")
        # keys.extend("text_embed")

        # tạo một từ điển ánh xạ các keys đến các chỉ số tương ứng của chúng 
        self.key_map = {key: i for key, i in enumerate(keys)}
        # gán các giá trị cho thuộc tính resampling và img_preproc 
        self.resampling = resample 
        self.img_preproc = img_preproc
        # kiểm tra xem liệu các URL có chứa liên kết đến Amazon S3 hay không 
        if (isinstance(urls, str) and "s3:" in urls) or (isinstance(urls, list) and any(["s3:" in url for url in urls])):
            # Then this has an s3 link for the webdataset and we need extra packages
            # nếu có sẽ kiểm tra xem công cụ s3cm có được cài đặt hay không
            if shutil.which("s3cmd") is None:
                # nếu không một lỗi runtime sẽ được ném ra 
                raise RuntimeError("s3cmd is required for s3 webdataset")
        
        # nếu như URL của thư mục chứa các embedding ảnh hoặc văn bản có chứa liên kết đến amazon s3
        if (img_embedding_folder_url is not None and "s3:" in img_embedding_folder_url) or (text_embedding_folder_url is not None and "s3:" in text_embedding_folder_url):
            # Then the embeddings are being loaded from s3 and fsspec requires s3fs
            try:
                # sau đó sẽ import một thư viện s3fs 
                import s3fs
            # nếu như thư viện này không được cài đặt, một lỗi RuntimeError sẽ được ném ra 
            except ImportError:
                raise RuntimeError("s3fs is required to load embeddings from s3")
        
        
        # Add the shardList and randomize or resample if requested
        # thêm một danh sách các shard vào dối tượng hiện tại 
        if resample:
            # nếu resample = True nó sẽ thêm một đối tượng ResampledShards 
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        # nếu không 
        else:
            # thêm một đối tượng SimpleShardList 
            self.append(wds.SimpleShardList(urls))
            # và nếu shuffle_shards = True nó cũng sẽ thêm một bộ lọc shuffle 
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        
        # nếu như embedding text URL tồn tại 
        if img_embedding_folder_url is not None :
            # hàm skip_unassociated_shards được gọi để lấy các file embedding tương ứng với các shard trong url 
            # kết quả của hàm này sẽ được append vào pipeline 
            self.append(skip_unassociated_shards(embeddings_url=img_embedding_folder_url, handler=handler))
        # tươnh tự như trên ta áp dụng đối chiếu với img 
        if text_embedding_folder_url is not None:
            # tương tự như trên gọi hàm skip_unassociated_shards bỏ qua các shards không có embedding
            # tương ứng kết quả cuối cùng cũng được thêm vào pipeline 
            self.append(skip_unassociated_shards(embeddings_url=text_embedding_folder_url, handler=handler))

        # hàm wds.tarfile_to_sample được gọi để chuyển đổi các tập tin tarfile thành các 
        # từ điển sample (các mẫ)
        self.append(wds.tarfile_to_samples(handler=handler))
        # Hàm wds.decode được gọi để giải mã dữ liệu sử dụng decode "pilrgb"
        self.append("pilrgb", handler=handler)
        # nếu img_embedding_folder_url không phai none 
        if img_embedding_folder_url is not None: 
            # sau đó tải các ảnh nhúng cho một nguồn từ xa 
            # đảm bảo rằng chỉ số index_width tồn tại để lấy các độ dài phần nhúng riêng biệt theo chỉ mục 
            assert index_width is not None, "Reading embeddings separately requires index width length to be given"
            #sau khi tách phần nhúng tương ứng thêm kết quả vào pipeline để thêm các embedding vào 
            # các mẫu dữ liệu
            self.append(insert_embedding(embeddings_url=img_embedding_folder_url, index_width=index_width, sample_key='img_emb', handler=handler))

        # tương tự nếu như text_embedding không phải None 
        if text_embedding_folder_url is not None: 
            # sau đó tải các ảnh nhúng cho một nguồn từ xa 
            # đảm bảo rằng chỉ số index_width tồn tại để lấy các độ dài phần nhúng riêng biệt theo chỉ mục 
            assert index_width is not None, "Reading embeddings separately requires index width length to be given"
            #sau khi tách phần nhúng tương ứng thêm kết quả vào pipeline để thêm các embedding vào 
            # các mẫu dữ liệu
            self.append(insert_embedding(embeddings_url=text_embedding_folder_url, index_width=index_width, sample_key='text_emb', handler=handler))
        
        # hàm join_embeddings và key_verify(required_keys = keys) đươc thêm vào 
        # pipeline 
        self.append(join_embeddings)
        self.appdend(verify_keys(required_keys=keys, handler=handler))

        # áp dụng tiền xử lý sử dụng hàm wds.map để áp dụng tiền xử lý preprocessing lên mẫu 
        self.append(wds.map(self.preproc))
        # cuối cùng là thêm một danh sách các keys vào pipeline chúng được 
        # chuyển đổi dnagj tuple 
        self.append(wds.to_tuple(*keys))

    
    # Thiết lập phưuong thức def preproc để áp dụng việc xử lý trước cho các hình ảnh 
    def preproc(self, sample):
        """Applies the preprocessing for images"""
        # kiểm tra thuộc tính self.img_preproc có = None 
        if self.img_preproc is not None : 
            # nếu tham số này = True 
            # trích xuất danh sách các ảnh có định dạng jpg trong từ điển sample 
            # và áp dụng lên các ảnh này tiền xử lý trước 
            sample["jpg"] = self.img_preproc(sample["jpg"])
        return sample
    

# khởi tạo phương thức tải dữ liệu hình ảnh đã được nhúng 
def create_image_embedding_dataloader(
    tar_url,
    num_workers,
    batch_size,
    img_embeddings_url=None,
    text_embeddings_url=None,
    index_width=None,
    shuffle_num = None,
    shuffle_shards = True,
    resample_shards = False, 
    img_preproc=None,
    extra_keys=[],
    handler=wds.handlers.reraise_exception#warn_and_continue
):
    """
    Convenience function to create an image embedding dataseta and dataloader in one line

    :param tar_url: A url pointing to the tar files of the webdataset formatted as /path/to/webdataset/{0000..9999}.tar
    :param num_workers: The number of workers to use for the dataloader
    :param batch_size: The batch size to use for the dataloader
    :param embeddings_url: Required if webdataset does not contain embeddings. A url pointing to the npy files of the embeddings. Should have the same number of shards as the webdataset.
        Webdataset image keys should align with the index of the embedding. This means missing image indices must have a corresponding embedding of all zeros.
    :param index_width: The number of digits in the index. This is used to align the embedding index with the image index.
            For example, if a file in the webdataset shard 3 is named 0003039.jpg, we know the shard is 4 digits and the last 3 digits are the index_width.
    :param shuffle_num: If not None, shuffle the dataset with this size buffer after sampling.
    :param shuffle_shards: If true, shuffle the shards before sampling. This cannot be true if resample is true.
    :param resample_shards: If true, resample webdataset shards with replacement. You need to set your own epoch size if this is true since it will resample infinitely.
    :param handler: A webdataset handler.
    """
    ds = ImageEmbeddingDataset(
        tar_url,
        img_embedding_folder_url=img_embeddings_url,
        text_embedding_folder_url=text_embeddings_url,
        index_width=index_width,
        shuffle_shards=shuffle_shards,
        resample=resample_shards,
        extra_keys=extra_keys,
        img_preproc=img_preproc,
        handler=handler
    )
    if shuffle_num is not None and shuffle_num > 0:
        ds.shuffle(1000)
    return DataLoader(
        ds,
        num_workers=num_workers,
        batch_size=batch_size,
        prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
        pin_memory=True,
        shuffle=False
    )