import os 
from logging import getLogger 
from pathlib import Path 
from typing import (
    AbstractSet, 
    cast, 
    Collection, 
    Dict, 
    Iterator, 
    List, 
    Literal, 
    Sequence, 
    TypedDict, 
    Union, 
)

import tiktoken 
from tiktoken.load import load_tiktoken_pbe 

# Định nghĩa một logger 
logger = getLogger(__name__) # logger này được sử dụng để lưu trữ và in một thông tin Info

# Sử dụng hàm Literal để chỉ định các giá trị cho biến Role 
Role = Literal["system","user","assistant"]

# Xây dựng lớp Message được định nghĩa như là một kiểu từ điển 
class Message(TypedDict):
    # Gồm 2 keys là role và content 
    # role có đối số là Role 
    role : Role 
    content : str 

# Khởi tạo Sequence[Mesage] cho phép khai báo một danh sách đối tượng Message
# Khi gán giá trị vào biến có kiểu Dalog nó chỉ có thể lấy các phần tử kiểu Message 
# đã định nghĩa 
Dialog = Sequence[Message]

# Xây dựng lớp ãm hóa mã thông báo tokenizer 
class Tokenizer: 
    """
    Mã hóa và mã hóa / giải mã văn bản sử dụng mã hóa mã thông báo TiToken 
    """

    # Xây dựng từ điển special với 2 giá trị là str và int 
    # biểu thị cho biết rằng chỉ có thể lấy các keys là str và int 
    special_tokens: Dict[str, int]

    # Biến num_preserved_special_tokens chỉ định cho biết số lượng 
    # token đặc biệt được bảo trì = 256
    num_preserved_special_tokens = 256 

    # Biến pat_str là một chuỗi regular experision (nghiên cứu regex) dùng để định dnagj và 
    # tokenize văn bản. Chuỗi này sẽ chia nhỏ văn bản thành các phần tử nhỏ 
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501


    # Xây dựng phương thức init phương thức này sẽ thực hiện khởi tạo mô hình 
    # mã hóa mã thông báo 
    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file
        """
        # đảm bảo rằng đầu vào model_path là một đường dẫn thư mục
        # nếu không tồn tại thì một lỗi sẽ được ném ra đồng thời hiển thị giá trị của model_path  
        assert os.path.isfile(model_path), model_path 

        # Gọi hàm load_tiktoken_bpe để tải mô hình bpe  được đào tạo trước  mô hình này xác định các 
        # từ phụ có thể được hợp nhất thành các từ lớn hơn 
        mergeable_ranks = load_tiktoken_pbe(model_path)
        # sau đó tính toán độ dài của dnah sách này 
        num_base_tokens = len(mergeable_ranks)

        # Xác định một danh sách các token đặc biệt được sử dụng bởi mô hình 
        # các token này sẽ được sử dụng để đánh dấu bắt đầu và kết thúc của văn bản 
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            # Danh sách này sẽ được cộng với 1 danh sách khác 
            # Là một list Comprehension được sử dụng để tạo ra thêm các  special_tokens 
            # dựa trên lượng token đặc biệt dành riêng đã được xác định 
            f"<|reserved_special_token_{i}|>"
            # nó bắt đầu từ tokem thứ 5 và kết thúc trước token cuối cùng của phạm vi đã định 
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]

        # Khởi tạo một từ điển sprecial_tokens nơi mỗi token đặc biệt từ danh sách 
        # sprecial_tokens được gán mọt giá trị số duy nhất không trùng lặp với bất 
        # kỳ token cơ bản naoò 
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        # Khởi tạo mô hình Tiktoken.Encoding đối tượng này chứa các thông tin cần thiết để mã hóa và giải 
        # mã các token khi huấn luyện hoặc sử dụng mô hình 
        self.model = tiktoken.Encoding(
            # name tên của mô hình lấy từ tên file trong đường dẫn model_path 
            #
            name = Path(model_path).name, 
            # pat_str một chuỗi mẫu (pattern string) được sử dụng trong qúa trình mã hóa 
            pat_str= self.pat_str, 
            # mergeable_ranks danh sách các cặp token có thể được ghép nối từ BPE 
            mergeable_ranks=mergeable_ranks, 
            # special_tokens Dictionary các token đặc biệt đã được 
            # thiết lập 
            special_tokens=special_tokens
        )

        # Sử dụng trình ghi nhật ký logger để ghi lại thông báo rằng mô hình 
        # Tiktoken đã được tải lại từ đường dẫn model_path
        logger.info(f"Reloaded tiktoken model from {model_path}")

        # Lấy ra kích thước của tập từ điển là số lượng 
        # từ mà tập từ điển có thể chứa đựng 
        self.n_words: int = self.model.n_vocab 

        # Xác định Id cho các token đặc biệt và thiết lập các điều kiện dừng cho quá trình 
        # sinh token 
        # slef.bos_id ID của token đánh dấu bắt đầu của văn abnr  nó được lấy từ self.special_tokens 
        # đã được thiết lập trước đó 
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        # self.eos_id Là ID của token đánh dấu kết thúc của văn bản 
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        # self.pad_id đưicj sử dụng để đẹm padding các chuỗi văn bản để chúng có cùng độ dài khi xử lý 
        # trong trường hợp anyf giá trị -1 có thể đại diện cho cho việc không sử dụng token 
        # đệm nào hoặc có thể là một giá trị đặc biệt được mô hình xử lý riêng 
        self.pad_id = -1 
        # self.stop_token đây là một tập hợp các token mà khi mô hình gặp phải 
        # sẽ dừng sinh token. Nó bao gômg token đánh dấu kết thúc văn bản 
        # và token đánh dấu kết thúc lượt 
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"]
        }

        # cuối cùng một thông báo được ghi lại bởi logger thông báo số lươngh 
        # từ trong mô hình self.n_words cùng với ID Của BOS và EOS 
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
    

    # Xây dựng phương thức encode để giải mã các mã thông báo thành các token ids 
    def encode(self, s: str, *, bos: bool, eos: bool, 
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),   
               )-> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """

        # đảm bảo rằng kiểu dữ liệu của s là string 
        assert type(s) is str 

        # Trình mã hóa mã thông báo Tiktoken có thể xử lý khoảng 400 ngàn ký tự 
        # Vậy nên ta định nghĩa một hằng số đại diện cho số lượng ký tự tối đa cho tiktoken 
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000 

        # Vì công việc thực hiện lặp lại các đoạn và tách chúng nếu như chúng không vượt quá giới hạn 
        # tối đa của các ký tự không phải khoảng trắng hoặc các khoảng trắng liên tiếp 
        # được giới hạn trong ngưỡng 25000
        MAX_NO_WHITESPACES_CHARS = 25_000 

        # Sử dụng một generator expression tạo một chuỗi các substr chuỗi con từ chuỗi văn bản đầu vào x 
        substrs = (
            substr 
            # sau đó lặp qua một dnah sách từ 0-> độ dài của s với sải bước TIKTOKEN_MAX_ENCODE_CHARS 
            # là số lượng ký tự tối đa mà mỗi chuỗi con có thể chứa 
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            # HÀM self._split_whitespaces.. được gọi để chia chuỗi s thành các chuỗi con dựa trên khoảng trắng 
            # các ký tự không phải khoảng trắng, với mục đích giữ nguyên cấu trúc văn bản 
            # các chuỗi con này sẽ có độ dài 25_000
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )

        # KHỞI TẠO MỘT DANH SÁCH T DANH SÁCH NÀY CHỈ CHỨA DỮ LIỆU LÀ INT 
        t: List[int] = []
        # duyệt qua dnah sách chứa các đoạn văn bản con kích thước 25_000
        for substr in substrs: 
            # Kêu gọi hàm extend để thực hiện ghép nối các khối văn bản đượcn
            # giải mã 
            t.extend(
                # kêu gọi phương thức self.model.encode để giải mã các tokens thành Token IDS 
                self.model.encode(
                    substr, 
                    # tham số allowed_special cho phép các token đặc biệt được phép trong chuỗi 
                    allowed_special= allowed_special, 
                    # và disallowed_special không cho phép các tokens đặc biệt không thể giải mã có trong chuỗi 
                    disallowed_special= disallowed_special, 
                )
            )

        # kiểm tra xem tham số bos tham số này chỉ định xem có thêm token đánh 
        # dấu vị trí bắt đầu văn bản hay không 
        if bos: 
            # nếu có thực hiện trèn bos_id vào vị trí bắt đầu 
            t.insert(0, self.bos_id)
        
        # Đồng thời cũng phải kiểm tra xem tham số eos tham số này chỉ định xem có thêm token đánh 
        # đấu vị trí kết thúc văn bản hay không 
        if eos: 
            # Thêm token này vào vị trí kết thúc 
            t.append(self.eos_id)
        
        # Cuối cùng trả về danh sách token IDS t đã được xử lý 
        return t 
    

    # Xây dựng phương thức decode phương thức này được sử dụng để giải mã văn bản 
    # từ một danh sách các tokens IDS thành một chuỗi văn bản 
    def decode(self, t: Sequence[int]) -> str: 
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        # kêu gọi phương thức self.model.deocde để giải mã các list[int] token ids 
        # hàm cast được sử dụng trong quá trình giúp chuyển đổi kiểu dữ liệu của List[int] cho t 
        return self.model.decode(cast(List[int], t))
    

    # định nghĩa một phương thức tĩnh với @staticmethod 
    # phương thức này sẽ không nhận bất kỳ tham số ngầm định nào liên quan đến instance cụ thể 
    # của lớp như self hoặc lớp đó như cls 
    # phương thức _split_whitespcaes_or_nonwhitespaces được sử dụng để 
    # chia một chuỗi đầu vào thành cac chuôi con không chứa nhiều hơn max_consecutive_slice_len 
    # ký tự liên tiếp giống nhau 
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:

        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        # curren_slice_len khởi tạo độ dài của chuỗi con hiện tại 
        current_slice_len = 0
        # kiểm tra xem ký tự đầu tiên của chuỗi s có phải là khoảng trắng hay không với hàm isspace
        # và gán giá trị tương ứng cho curren_slice_is_space 
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        # Đặt chỉ số đầu của chuỗi con hiện tại là 0
        slice_start = 0

        # vòng lặp for duyệt qua từng ký tự trong chuỗi s 
        for i in range(len(s)):
            # kiểm tra xem ký tự hiện tại trong chuỗi có phải là khoảng trắng hay không 
            is_now_space = s[i].isspace()

            # Sử dụng toán tử xor để kiểm tra xem có sự thay đổi giữa ký tự khoảng trắng và không 
            # phải khoảng trắng 
            if current_slice_is_space ^ is_now_space: 
                # Đặt current_slice_len = 1 vì chuỗi mới bắt đầu 
                current_slice_len = 1 
                # Câp nhật current_slice_is_space với giá trị mới 
                current_slice_is_space = is_now_space
            # Nếu không có sự thay đổi tăng current_slice lên 1 
            else:
                # Tăng độ dài chuỗi con hiện tại lên 1 
                current_slice_len += 1
                # nếu độ dài chuỗi con hiện tại > độ dài tối đa cuar chuỗi con 
                if current_slice_len > max_consecutive_slice_len:
                    # sử dụng từ khóa yield  để trả về chuỗi con từ chỉ số slice_start đến i
                    yield s[slice_start:i]
                    # sau đó cập nhật lại chỉ số bắt đầu cho chuỗi con mới
                    slice_start = i
                    # sau đó cập nhật luôn độ dãi cho chuỗi con mới 
                    current_slice_len = 1
        
        # cuối cùng sau khi vòng lặp kết thúc thực hiện trả về chuỗi con cuối cùng 
        # từ slice_start đến hết chuỗi s
        yield s[slice_start:]



# Xây dựng lơp phương ChatFormat để định dạng một khung chat 
# cho mô hình 
class ChatFormat: 
    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính 
    def __int__(self, tokenizer: Tokenizer):
        # định nghĩa thuộc tính tokenizer 
        self.tokenizer = tokenizer 

    # Thiết lập phương thức encode_hander phương thức này ma xhoas phần đầu của tin nhắn 
    # và trả về một List[int] token_ids 
    def encode_header(self, message: Message) -> List[int]:
        # Khởi tạo một danh sách token = None các tokens đầu nội dung tin nhắn 
        # vào dnah sách 
        tokens = []
        # 1: Thêm token đăch biệt đánh dấu vị trí bắt đầu vào phần đầu tin nhắn 
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        # 2: Mã hóa vai trò role của người gửi tin nhắn mà không thêm token bắt đầu hoặc kết thúc
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        # 3: Thêm token đặc biệt để đánh dấu kết thúc phần đầu tin nhắn 
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        # Thêm mã hóa của 2 dòng mới để tách biệt phần đầu với phần nội dung tin nhắn 
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))

        # Cuối cùng trả về danh sách các phần đầu tin nhắ đã được mã hóa token ids
        return tokens
    
    # Xây dựng phương thức encode_message để mã hóa tin nhắn văn bản 
    # tức mã hóa toàn bộ tin nhắn 
    def encode_message(self, message: Message) -> List[int]:
        # Gọi phương thức encode_header để thực hiẹn mã hóa phần đầu tin nhắn 
        tokens = self.encode_header(message)
        # Tiếp theo mã hóa các phần nội dung tin nhăn 
        # hàm extend sẽ đựo sử dụng để nối các phần nội dung được mã hóa lại với nhau 
        tokens.extend(
            # gọi đến phương thức self.tokenizer.encode để mã hóa phần  nội dung tin nhắn 
            # hàm strip được gọi để bỏ qua các khoảng trắng và đồng thời các ký tự đánh dấu cũng sẽ bị bỏ đi 
            self.tokenizer.encode(message["content"].strip(), bos= False, eos=False)
        )

        # Thêm token đặc biệt để đánh dấu kết thúc tin nhắn 
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        # cuối cùng trả về danh sách các token message đã được mã hóa 
        return tokens
    

    # Xây dựng phương thức encode_dialog_prompt để thực hiện mã hóa một cuộc đối thoại 
    # thành một chuỗi số nguyên chuẩn bị cho việc đào tạo mô hình 
    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        # khởi tạo một danh sách các token để lưu trữ danh sách các token ids 
        tokens = []
        # Thêm token đặc biệt đánh dấu bắt đầu văn bản vào danh sách 
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        # duyệt qua toàn bộ tin nhắn trong hộp thoại 
        for message in dialog:
            # Mỗi tin nhắn trong hộp thoại được mã hóa bằng phương thức encode_message 
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        # cuối cùng phương thức này thêm một phần đầu tin nhắn của trợ lý assistant vào cuộc đối thoại 
        # nhưng nó không có nội dung để mô hình có thể hoàn thiện việc còn lại 
        # Điều này tạo ra một prompt cho mô hình, nơi mô hình được yêu cầu sinh ra phần tiếp theo 
        # của cuộc đối thoại dựa trên thông tin được mã hóa 
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        # Phương thức trả về danh sách tokens chứa chuỗi số nguyên đã mã hóa, sẵn sàng để được đưa vào mô hình học máy.
        return tokens
    
    # Như vậy, encode_dialog_prompt chuẩn bị dữ liệu đầu vào cho mô hình bằng cách mã hóa toàn bộ cuộc đối thoại và
    # tạo ra một điểm khởi đầu cho mô hình để sinh ra phản hồi tiếp theo.