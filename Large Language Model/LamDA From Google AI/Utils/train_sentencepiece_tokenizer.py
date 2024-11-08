from datasets import load_dataset 
# import sentencePieceBPETokenizer 
import io 
import sentencepiece as spm 

# load the dataset 
dataset = load_dataset('conceptofmind/pile_wikipedia_en', split='train', streaming=True)


# Thực hiện mã hóa BPE tokenizer 
# duyệt qua danh sách các tệp trong danh sách dataset 
def batch_iterator(dataset):
    for i in dataset: 
        # sử dụng yield để trả về giá trị của khóa text từ phần tử i 
        yield i['text']

# thiết lập kiểu định dạng mã hóa sử dụng mã hóa bytes 
model = io.BytesIO()

# thực hiện mã hóa BPE cho danh sách dữ liệu 
spm.SentencePieceTrainer.train(
    # lấy ra các text trong các tệp văn bản 
    sentence_iterator = batch_iterator(dataset), 
    # một định dạng mã hóa byte
    model_writer=model, 
    # kích thước tập từ điển = 32 000
    vocab_size=32000,
    # phương pháp mã hóa Byte pair encoding 
    model_type='bpe',
)
