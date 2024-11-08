# take from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
# to give users a quick easy start to training DALL-E without doing BPE 

import torch 
import html 
import os 
import ftfy 
import regex as re 
from functools import lru_cache 
from pathlib import Path 

from dalle2_pytorch.utils import import_or_print_error

# định nghĩa một decorattor @lru_cache sử dụng để tối ưu hóa 
# việc lưu trữ kết quả. Đây là một cơ chế lưu chữ tạm thời cho các giá trị đã tinh 
# toán trước đó, giúp tránh việc tính toán lại các giá trị đã được lưu trữ trong cache 
@lru_cache()
def default_bpe():
    # trả về một đường dẫn tới tệp văn bản text_file có tên là 
    "bpe_simple_vocab_16e6.txt"
    # với hàm path.join được sử dụng để ghép nối 1 chuỗi đường dẫn từ hệ thồng 
    # os.path.dirname để lấy ra tên thư mục tuyệt đối 
    # path.abspath lấy ra đường dẫn tuyệt đối của tệp mã nguồn hiện tại
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/bpe_simple_vocab_16e6.txt")



# tương tự như trên định nghĩa một decorator @lru_cache để lưu trữ lại kết quả của một phép 
# tính đảm bảo rằng tránh việc tính toán lại các giá trị đã được lưu trữ 
@lru_cache()
# định nghĩa một hàm tính toán mã hóa Bytes to  unicode 
def bytes_to_unicode():
    # khởi tạo một biến bs là một dnah sách các giá trị bytes. Nó bắt đầu từ giá trị của ký tự "!"
    # và kết thúc ở giá trị cảu ký tự "~", bao gồm các ký tự có giá trị byte từ ord("i") đến 
    # ord("¬") và từ ord("®") đến ord("ÿ"). Đây là các ký tự có thể in được trong bảng mã ASCII mở rộng.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    # sao chép lại toàn bộ dnah sách này và gán cho cs 
    cs = bs[:]

    # khởi tạo một biến đêm n = 0
    n = 0 
    # duyệt qua danh sách ký tự bytes có thể có từ 0 - > 256
    for b in range(2**8):
        # nếu như b không có trong danh sách bs 
        if b not in bs: 
            # thêm b vào dnah sách bs 
            bs.append(b)
            # thêm mọt giá trị mới vào danh sách cs bắt đầu từ  256 + n và tăng dần 
            cs.append(2 ** 8 + n)

            n += 1 
    
    # chuyển đổi tất cả các ký trong danh sách cs sang ký tự unicode 
    cs = [chr(n) for n  in cs]
    # tạo một từ điển dict ánh xạ từng ký tự byte trong bs sang ký tự unicode tương ứng trong cs 
    return dict(zip(bs, cs))


# Thiết lập phương thức get_pairs để lấy ácc cặp từ 
def get_pairs(word):
    # khởi tạo một danh sách pairs là một set
    # đảm bảo các cặp ký tự là duy nhất và không có thứ tự 
    pairs = set()
    # gán cho ký tự đúng trước tại chỉ số [0]
    prev_char = word[0]
    # duyệt qua danh sách bắt đầu từ chỉ số thứ 2 đến cuối 
    for char in word[1:]:
        # thêm các cặp ký tự tương ưng vào pairs 
        pairs.add((prev_char, char))
        # sau đó gán lại prev char = ký tự char hiện tại của vòng lặp
        prev_char = char # để có thể tiến hành lấy cặp tiếp theo 
    
    return pairs 


# Thiết lập phương thức basic_clean để thực hiện công việc dọn dẹp văn bản 
def basic_clean(text):
    # sử dụng hàm ftfy để sửa các lỗi mã hóa văn bản VD có thể chuyển các mã UNicode không đúng 
    # thành mã Unicode đúng 
    text = ftfy.fix_text(text)
    # Giải mã các ký tự HTML đặc biệt trong văn bản hàm unescape giải mã 1 lần 
    # sau đó thêm lần nữa để đảm bảo rằng tất cả các ký tự html đã được giải mã 
    text = html.unescape(html.unescape(text))
    # laoij bỏ các khoảng trắng đầu và cuối văn bản 
    return text.strip()

# whitescape_clean làm sạch các khoảng trắng trong văn bản 
def whitespace_clean(text):
    # sử dụng biểu thức chính quy regex để thay thế tất cả các khoảng trắng liên tiếp 
    # bằng một khoảng trắng duy nhất 
    text = re.sub(r'\s+', ' ', text)
    # Loại bỏ khoảng trắng đầu và cuôi văn bản 
    text = text.strip()
    # trả về văn bản đã được làm sạch 
    return text

# Thiết lập lớp phương thức simpleTokenizer để mã hóa các token văn bản 
class SimpleTokenizer(object):
    # Định nghĩa một phuong thức khởi tạo 
    def __init__(self, bpe_path= default_bpe()):
        # Khởi tạo một từ điển ánh xạ từ giá trị byte sang ký tự Unicode bằng cách sử dụng hàm bytes_to_uincode 
        self.byte_encoder = bytes_to_unicode()
        #  Tạo một từ điển ánh xạ ngược ký tự unicode sang Bytes
        self.byte_encoder = {v: k for k, v in self.bytes_to_unicode.items()}
        # Đcoj nội dung tệp văn bản tại đường dẫn bpe_path và chia thành danh sách các chuỗi
        # dựa trên dấu xuống dòng 
        merges = Path(bpe_path).read_text(encoding='utf8').split('\n')
        # khởi tạo danh sách merges là phần còn lại từ chỉ số 1 đến 49152 - 255 
        merges = merges[1:49152 - 256 - 2 + 1]
        # chuyển đổi mỗi chuỗi trong danh sách merges thành một tuple bằng cách tách các từ trong chuỗi 
        merges = [tuple(merge.split()) for merge in merges]
        # tạo danh sach vocab từ các giá trị trong từ điển byte_to_unicode 
        vocab = list(bytes_to_unicode().values())
        #  Thêm vào danh sách vocab các từ có thêm ký tự '</w>' ở cuối.
        vocab = vocab + [v + '</w>' for v in vocab]
        # duyệt qua các từ trong danh sách merges 
        for merge in merges:
            # và thêm các từ trong danh merge vào từ điển vocab 
            vocab.append(''.join(merge))
        # sau đó thêm các chuỗi đánh dâu bắt đầu và 
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        # định nghĩa kích thước của voab 
        self.vocab_size = 49408
        # tạo một từ điển ãnh xạ các từ trong từ điển sang chỉ số tương ứng 
        self.encoder = dict(zip(vocab, range(len(vocab))))
        # tạo một từ điển ánh xạ ngược chỉ số sang token trong từ điển 
        self.decoder = {v: k for k, v in self.encoder.items()}
        # tạo một từ điển ánh xạ các cặp từ trong merges sang các chỉ số tương ứng 
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        # Xây dựng biên dịch cho một loạt các token đặc biệt 
        # sử dụng biên dịch bởi biểu thức chính quy regex
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)
        

        # Thiết lập phương thức mã Hóa BPE 
        def bpe(self, token):
            # kiểm tra xem danh sách các token này có được lưu trữ trong bộ nhớ cache hay không 
            if token in self.cache: 
                # nêu như token được lưu trữ trong cache 
                # trả về danh sách các token được lưu chữ trong cache 
                return self.cache[token]
            
            # khởi tạo một tuple word tuple này chứadanh sách là danh sách các token đến chỉ số n - 1 và + danh sách thứ 2 
            # token cuối cùng và thêm vào ký tự cuối cùng với '</w>'
            word = tuple(token[:-1] ) + (token[-1])
            # khởi tạo một danh sách pairs để lưu trữ các cặp từ liên tiếp 
            # sử dụng hàm get_pairs để lấy ra các cặp từ liên tiếp từ danh sách token word
            pairs = get_pairs(word)

            # Nếu như danh sách pairs không tồn tại 
            if not pairs: 
                # trả về danh sách token + ký tự đánh dấu '</w>' 
                return token + '</w>'
            
            # Bắt đầu một vòng lặp vô hạn 
            while True : 
                # chọn một cặp ký tự có chỉ số thấp nhất dựa trên thứ tự ưu tiên từ self.bpe_ranks
                # nếu cặp không có trong self.bpe_ranks thì thoát khỏi vòng lặp 
                # sử dụng các pairs để tách các giá trị và tính toán chỉ số 
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
                # nếu cặp không có trong self.bpe_ranks 
                if bigram not in self.bpe_ranks: 
                    # thoát khỏi vòng lặp 
                    break 
                
                # gán giá trị của các cặp ký tự cho first và second 
                first, second = bigram
                # khởi tạo một danh sách new_word để lưu trữ các từ mới thỏa mãn 
                # với trình mã hóa bpe 
                new_word = []
                # và một biến i để đánh dấu 
                i = 0 
                # Trong khi mà chỉ số i còn nhỏ hơn độ dài của danh sách word 
                while i < len(word):
                    # xây dựng một khối ngoại lệ để đảm bảo nó sẽ luôn được thực thi 
                    # khi trương trình xảy ra một số lỗi vẫn có thể hoạt động để phương thức mã háo 
                    # BPE có thể thực thi việc ghép nối mà không gặp bất kỳ gián đoạn nào 
                    try: 
                        # tìm vị trí đầu tiên của firt trong danh sách word từ vị trí i 
                        j = word.index(first, i)
                        # Thêm vào danh sách new_word các giá trị từ i đến j 
                        new_word.extend(word[i:j]) # điều này đảm bảo ta thực hiện việc lấy mẫu rời rạc từng phần tử 1
                        # sau đó ta cập nhật lại i hiện tại để thực hiện lấy cặp tiếp theo 
                        i = j 
                    
                    # bắt đầu một khối except để xử lý các ngoại lệ được ném ra từ try 
                    except: # nếu không tìm thấy first thêm tất cả các ký tự từ i đến cuối dnah sách word 
                        # vào danh sách new_word 
                        new_word.extend(word[i:])
                        # kết thúc vòng lặp có nghĩa là không tìm thấy phần tử first trong bộ từ điển word 
                        break 
                    
                    # kiểm tra xem word[i] và word[i+1] có tạo thành cặp firt + second hay không 
                    # sau khi 2 ký tự liên tiếp được ghép nối và thỏa mãn thì ta tính toán đến trường hợp đặt lại 
                    # i là 1 cặp đã được ghép nối có nghĩa là bây giờ i = first mà first = [i:j] đã được hợp nhất bây giờ i = first và second sẽ là chỉ số j 
                    if word[i] == first  and i < len(word) - 1 and word[i + 1] == second: 
                        # thêm một cặp firt và second vào danh sách new_word là kết quả của phép nối 2 ký tự liền kề trong danh sách 
                        # word 
                        new_word.append(first + second)
                        # tăng giá trị của i lên 2. Tức là cập nhật ví trí hiện tại của biến i trong bộ word 
                        # để ở lần thực thi tiếp theo bỏ qua 2 phần tử đã được ghép nối 
                        i += 2 
                    
                    # trường hợp còn lại thêm vào danh sách new_word phần tử theo vị trí i hiện tại 
                    else: 
                        new_word.append(word[i])
                        # và tăng chỉ số i lên 1 
                        i += 1 
                
                # sau đó chuyển đổi danh sách new_word đã được ghép nối ở các bước trên thành 1 tuple 
                # để đảm bảo sự cố định và duy nhất của mỗi phần tử có trong danh sách 
                new_word = tuple(new_word)
                # gán lại cho danh sách word = tuple new_word 
                word = new_word
                # kiểm tra xem độ dài của danh sách này nếu như nó bằng 1 thì kết thúc vòng lặp 
                if len(word) == 1:
                    break 
                # trường hợp còn lại 
                else: 
                    # gọi đến hàm get_pairs để tạo ra một danh sách các cặp ký tự 
                    # liên tiếp mới từ word. Các phần tử đã được ghép nối trước đó bây giờ sẽ được tính là 
                    # một phần tử đơn lẻ 
                    pairs = get_pairs(word)
                
            # thêm các khoảng trắng ngăn cách các phần tử token đã được mã hóa BPE 
            word = ' '.join(word)
            # lưu danh sách word vào bộ nhớ cache 
            self.cache[token] = word 
            # trả về danh sách word đã được mã hóa 
            return word 
    
    
    # Thiết lập phương thức mã hóa encode nhận đầu vào là một chuỗi văn bản và trả về các token_ids (int)
    # biểu diễn mã hóa BPE đầu vào 
    def encode(self, text):
        # KHỞI TẠO  một danh sách để lưu trữ các token_ids được mã hóa 
        bpe_tokens = []
        # thực hiện loại bỏ các khoảng trắng và chuyển văn bản đầu vào thành chữ thường
        text = whitespace_clean(basic_clean(text).lower())
        # sử dụng biểu thức chính quy self.pat để tách văn bản thành các token 
        # hàm re.finnally có chức năng tìm kiếm tất cả khớp với biểu thức chính quy 
        for token in re.findall(self.pat, text): 
            # Mã hóa mỗi token trong chuỗi thành UTF-8 và ánh xạ từng bytess sang ký tự unicode 
            # tương ứng 
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # mở rộng danh sách bpe_tokens bằng cahc thêm các token được mã hóa BPE 
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        
        # kết quả trả về một danh sách chứa các chỉ số của các tokens được giải mã 
        # bằng cách ánh xạ các tokens bpe sang một chỉ số tương ứng trong từ điển vocab 
        return bpe_tokens
    

    # Thiết lập phương thức giải mã xử lý một quy trình ánh xạ ngược từ các chỉ số token 
    # thành văn bản 
    def decode(self, tokens, remove_start_end = True, pad_tokens = set()):
        # kiểm tra xem danh sách tokens có phải là một tensor hay không 
        if torch.is_tensor(tokens):
            # nếu nó là một tensor chuyển đổi nó thành 1 danh sahc s
            tokens = tokens.tolist()

            # loại bỏ đi các token đặc biệt chỉ đầu và cuối văn bản nếu cần 
            # và các tokens đệm khác 
            if remove_start_end: 
                # loại bỏ đi các token đánh dấu 
                token = [token for token in tokens if token in (49406, 40407, 0)]

            # thực hiện ánh xạ ngược các chỉ số tokens đã được mã hóa để lấy ra các tokens biểu 
            # diễn trong văn bản cụ thể 
            text = ''.join([self.decoder[token] for token in tokens if token not in pad_tokens])
            # tạo ra một bytearray từ một chuỗi text. Mỗi ký tự trong text được ánh xạ ngược từ chỉ số 
            # unicode sang mã byte tương ứng bằng cách sử dụng từ điển self.byte_deocder 
            # Sau đó chuyển đổi các chuỗi bytearray thành chuỗi văn bản bằng cách giải mã UTF-8 
            # và thay thế tất cả các ky tự xuống dòng '</w>' thành khoảng trắng 
            text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')

            # cuối cùng trả về văn bản đã được chuyển đổi từ các token_id sang tokens_ids -> byte -> utf-8 -> text 
            return text 
        

    # Thiết lập phương thức tokenize 
    def tokenize(self, texts, context_length = 256, truncate_text= False):
        # kiểm tra xem text có phải dnagj str
        if isinstance(texts, str):
            # gán text = dnah sahcs text 
            texts = [texts]

        # duyệt qua danh sách texs và ánh xạ chúng thành các token_id (chỉ số token trong từ điển)
        all_tokens = [self.encode(text) for text in texts]

        # khởi tạo một tensor result bằng danh sách all_tokens 
        # mỗi tensor trong danh sách này = context_length 
        result = torch.zeros(size=len(all_tokens), out=context_length, dtype=torch.long)

        # duyệt qua danh sách chứa id của các token: lấy ra các tensor chỉ số i và id tươn ứng với chỉ só 
        for i , tokens in enumerate(all_tokens):
            # kiểm tra xem độ dài của mỗi danh sách token thứ i > 256 
            if len(tokens) > context_length: 
                # thực hiện cắt lại danh sách này 
                tokens = tokens[:context_length]
            
            # trường hợp còn lại 
            else: 
                # ném ra một lỗi RunTime với thông báo sau 
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            # Tính toán lại mỗi danh sách i trong danh sách result và chuyển đổi các dnah sách token con 
            # thành các tensor 
            result[i, :len(tokens)] = torch.tensor(tokens)

        # cuối cùng trả về danh sách chứa các tensor biểu diễn token_id của văn bản 
        return result

# định nghĩa một tokenizer 
tokenizer = SimpleTokenizer()

tokenizer = SimpleTokenizer()

# YTTM tokenizer # định nghĩa một lớp Youtokenizer 
class YttmTokenizer:
    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(self, bpe_path = None):
        # khởi tạo một đường dẫn bpe_path 
        bpe_path = Path(bpe_path)
        # và đảm bảo giằng đường dẫn này có tônd tại 
        assert bpe_path.exists(), f'BPE json path {str(bpe_path)} does not exist'

        # định nghĩa một youtokentome 
        self.yttm = import_or_print_error('youtokentome', 'you need to install youtokentome by `pip install youtokentome`')

        # ĐỊnh ngĩa YoutokenTome với mô hình ma hóa BPE 
        tokenizer = self.yttm.BPE(model = str(bpe_path))
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size()

    # decode giải mã các token id thành văn bản có thể đọc được 
    def decode(self, tokens, pad_tokens = set()):
        # kiểm tra xem token có phải là một tensor
        if torch.is_tensor(tokens):
            # chuyển nó thành danh sách 
            tokens = tokens.tolist()

        # mã hóa các danh sahcs id token thành văn bản 
        return self.tokenizer.decode(tokens, ignore_ids = pad_tokens.union({0}))

    # hàm encode chuyển đổi văn bản thành các token mã hÓA id  
    def encode(self, texts):
        # sử dụng mô hình BPE để thực hiện mã hóa ID 
        encoded = self.tokenizer.encode(texts, output_type = self.yttm.OutputType.ID)
        # sử dụng map để áp dụng hàng loạt torch.tensor lên encoded 
        return list(map(torch.tensor, encoded))

    # okenize (tokenize): Hàm này chuyển đổi văn bản thành tensor của các token đã mã hóa
    def tokenize(self, texts, context_length = 256, truncate_text = False):
        # kiểm tra trường hợp văn bản  có dạng string 
        if isinstance(texts, str):
            texts = [texts]

        # chuyển đổi văn bản thành danh sách token_id 
        all_tokens = self.encode(texts)

        # tạo một danh có kích thước tương tự all_token mỗi 
        # danh sách con = context_length 
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            # đảm bảo rằng độ dài dnah sách con không vượt quá 256 
            if len(tokens) > context_length:
                # thực hiện cắt bớt về tiêu chuẩn 256 nếu nó vượt quá 
                if truncate_text:
                    tokens = tokens[:context_length]
                # trường hợp còn lại 
            else: 
                # ném ra một lỗi RunTime với thông báo sau 
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            # Tính toán lại mỗi danh sách i trong danh sách result và chuyển đổi các dnah sách token con 
            # thành các tensor 
            result[i, :len(tokens)] = torch.tensor(tokens)

        # cuối cùng trả về danh sách chứa các tensor biểu diễn token_id của văn bản 
        return result
