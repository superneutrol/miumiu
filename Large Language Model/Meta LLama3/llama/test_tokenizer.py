import os 
from unittest import TestCase 
from llama.tokenizer import ChatFormat, Tokenizer 

# Xây dựng lớp phương thức TokenizerTest để kiểm tra các thành phần mềm để 
# đảm bảo chúng hoạt đoọng như mong đợi. Lớp này kế thừa từ TestCase của thư viện unitest
# cho phéo định nghĩa accs phương thức kiểm tra khác nhau 
class TokenizerTests(TestCase):
    # Xây dựng phương thức setup được gọi trước mỗi phương thức kiểm tra 
    # Nó thiết lập môi trường kiểm tra bằng cách khởi tạo Tokenizer 
    def setUp(self):
        # khởi tạo thuộc tính tokenizer với đường dẫn được lấy từ biến môi trường 
        # Tokenizer_Path 
        self.tokenizer = Tokenizer(os.environ["TOKENIZER_PATH"])
        # Định nghĩa thuộc tính chat form là một lớp được sử dụng để định dạng tin nhắn chat 
        # thực hiện học các mã hóa tin nhắn văn bản 
        self.format = ChatFormat(self.tokenizer)

    # Xây dựng phương thức test_special_tokens phương thức này được sử dụng để 
    # kiểm tra một giá trị cho token đặc biệt 
    def test_special_tokens(self):
        # phương thức self.assertEqual sử dụng để kiểm tra xem token đánh dấu bắt 
        # đầu của văn bản có giá trị ID  mong đợi là 128000 hay không 
        self.assertEqual(
            self.tokenizer.special_tokens["<|begin_of_text|>"],
            128000,
        )

    # Xây dựng phương thức test_encode
    def test_encode(self):
        # phương thức self.assertEqual được sử dụng để kiểm tra xemm
        self.assertEqual(
            # phương thức self.tokenize.encode có mã hi=óa câu "This is a test sentence"
            # Thành một danh sách token IDS tươnh ứng hay không   
            self.tokenizer.encode(
                "This is a test sentence.",
                bos=True,
                eos=True
            ),
            [128000, 2028, 374, 264, 1296, 11914, 13, 128001],
        )
    
    # Xây dựng phương thức test_encode_message phương thức này có chức năng thực 
    # nghiệm giải mã các tin nhắn văn bản 
    def test_encode_message(self):
        # Xây dựng một từ điển message từ điển này gồm 
        # "role": một keys với values là văn bản sẽ cho biết vai trò của văn bản 
        # hoặc vai trò của bên đối thoại 
        # "content": có value thể hiện nội dung văn bản 
        message = {
            "role": "user",
            "content": "This is a test sentence.",
        }
        # Sử dụng hàm self.assertEqual để kiểm tra xem 
        # các token đặc biệt được mã hóa theo đoạn tin nhắn có bằng với cac giá trị kỳ vọng 
        # hay không 
        self.assertEqual(
            # Hàm self.format.encode_message được gọi để giải mã đoạn hội thoại tin nhắn
            self.format.encode_message(message),
            [   128006,  # <|start_header_id|>
                882,  # "user"
                128007,  # <|end_header_id|>
                271,  # "\n\n"
                2028, 374, 264, 1296, 11914, 13,  # This is a test sentence.
                128009,  # <|eot_id|>
            ]
        )

    

    # Xây dựng phương thức test_encode_dialog phương thức này được sử dụng để 
    # thử nghiệm giải mã một đoạn tin nhắn hội thoại 
    def test_encode_dialog(self):
        # Xây dựng một hộp thoại tin nhắn 
        # của system và người nhắc nhở 
        dialog = [
            {
                "role": "system",
                "content": "This is a test sentence.",
            },
            {
                "role": "user",
                "content": "This is response.",
            }
        ]
        # Hàm self.assertEqual để kiểm tra xem các ký tự đặc biệt trong chuỗi văn bản 
        # có được mã hóa thành các IDS kỳ vọng hay không  
        self.assertEqual(
            # Hàm self.format.encode_dialog_prompt  được sử dụng để giải mã một cuộc hội thoại văn bản 
            self.format.encode_dialog_prompt(dialog),
            [
                128000,  # <|begin_of_text|>
                128006,  # <|start_header_id|>
                9125,     # "system"
                128007,  # <|end_header_id|>
                271,     # "\n\n"
                2028, 374, 264, 1296, 11914, 13,  # "This is a test sentence."
                128009,  # <|eot_id|>
                128006,  # <|start_header_id|>
                882,     # "user"
                128007,  # <|end_header_id|>
                271,     # "\n\n"
                2028, 374, 264, 2077, 13,  # "This is a response.",
                128009,  # <|eot_id|>
                128006,  # <|start_header_id|>
                78191,   # "assistant"
                128007,  # <|end_header_id|>
                271,     # "\n\n"
            ]
        )