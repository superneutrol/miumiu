import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.distributed
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer


# Xây dựng lớp completionPrediction lớp này 
# được định nghĩa là một từ điển 
class CompletionPrediction(TypedDict, total=False):
    # key thứ nhất là generation có thể là một chuỗi văn bản được 
    # sinh ra theo dự đoan 
    generation: str
    # Token danh sách các mã thông báo 
    tokens: List[str]  # not required
    # logprob xác suất của nhật ký được dự đoán bởi mô hìn 
    logprobs: List[float]  # not required


# Tương tự như trên lớp ChatPrediction cũng được định nghĩa là từ điển 
# lớp này chứa các thông tin về chat văn bản 
class ChatPrediction(TypedDict, total=False):
    # Generation: Message các đoạn tin nhắn văn bản được tạo ra 
    generation: Message
    # Token danh sách các mã thông báo 
    tokens: List[str]  # not required
    # logprob xác suất của nhật ký được dự đoán bởi mô hìn 
    logprobs: List[float]  # not required


# Xây dựng lớp Generator Llama cho mô hình Meta-Llama 3 
class Llama:
    # xây dựng phương thức static là phupwng thức tĩnh của lớp Llama 
    # phương thức này được gọi để không cần phải sử dụng bất kỳ instance nào 
    # từ self hoặc cls của lớp 
    @staticmethod 
    def build(
        ckpt_dir: str, 
        tokenizer_path: str, 
        max_seq_len: int, 
        max_batch_size: int, 
        model_parallel_size: Optional[int] = None , 
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        # kiểm tra xem môi trường phân tán đã được thiết lập hay chưa 
        if not torch.distributed.is_initialized():
            # Nếu chưa được thiết lập ta thực hiện khởi tạo nó 
            # sử dụng torch.distributed.init_process_group để tạo các nhóm tiến trình trên 
            # môi trường phân tán 
            torch.distributed.init_process_group("nccl")
        
        # Kiểm tra xem mô hình sử lý song song đã được khởi tạo chưa 
        # mục đích xây dựng mô hình song song để có thể xử lý song song các nhóm GPU 
        # trong môi trương
        if not model_parallel_is_initialized():
            # Nếu chưa được khởi tạo ta tiến hành kiểm tra vê kích thước hay số lượng 
            # các tiến trình song song [mô hình song song] có tồn tại 
            if model_parallel_size is None: 
                # Nếu như số lượng các tiến trình song song không được định nghĩa 
                # ta thực hiện khởi tạo nó với kích thước = số GPU được sử dụng trong môi trường 
                # phân tán nếu không gán = 1
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            
            # Cuối cùng, hàm initialize_model_parallel(model_parallel_size) được gọi để khởi tạo mô hình song song với kích thước đã được xác định
            initialize_model_parallel(model_parallel_size)

        
        # Lấy ra thứ tự hoặc ID của các tiến trình trong môi trường phân tan 
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Thiết lập GPU dựa trên local_rank để đảm bảo mỗi quá trình sẽ sử dụng GPU 
        # tương ứng 
        torch.cuda.set_device(local_rank)

        # Đặt seed cho trình sinh số ngẫu nhiên của Pytorch để đảm bảo tính nhất quán trong các quá trình 
        torch.manual_seed(seed)

        # Nếu local_rank lớn hơn 0 chuyển hướng đầu ra chuẩn của quá trình đó đến  /dev/null
        # có nghĩa là không thể hiển thị đầu ra trên màn hình 
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        # Thiết đặt thời gian bắt đầu 
        start_time = time.time()
        # sử dụng hàm glob để tìm kiếm tất cả các tệp có đường dẫn .pth đực chỉ định 
        # sau đó xắp xếp chúng 
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        # Đảm bảo rằng số lượng tệp checkpoint > 0 
        assert checkpoints > 0, f"no checkpoint files found in {ckpt_dir}"
        # Kiểm tra xem số lượng checkpoint có khớp với mô hình song song hay không 
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        # Đặt các tệp checkpoint với các rank của mô hình song song 
        ckpt_path = checkpoints[get_model_parallel_rank()]
        # sau đó tải các tệp checkpoint với các chỉ số rank của mô hình song song 
        # hiện tại 
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # Sử dung hàm with để mở tệp Json có tên params.json kết quả gán cho biến f 
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            # sau đó sử dụng hàm json.loads để tải thông tin đọc đựoc từ tệp 
            # kết quả là một đối tượng python chứa thông tin của tệp thường sẽ là một từ điển 
            params = json.loads(f.read())

        
        # Khởi tạo một thực thể ModelArgs với các tham số khác được truyền vào thông qua từ 
        # điển param 
        model_args: ModelArgs = ModelArgs(
            max_seq_len= max_seq_len, 
            max_batch_size=max_batch_size, 
            **params,
        )
        # Thực hiện mã hÓA Tokenizer 
        tokenizer = Tokenizer( model_path= tokenizer_path)
        # Đảm bảo rằng kích thước của từ vựng trong model_args khớp với ố lượng 
        # từ trong tokenizer 
        assert model_args.vocab_size == tokenizer.n_words
        # Thiết lập tensor mặc dịnh kiểm tra xem GPU có hỗ trợ kiểu dữ liệu BFfloat 16
        if torch.cuda.is_bf16_supported():
            # vÀ THIết lập kiểu tensor mặc định tương ứng với BFloat16Tensor 
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        
        # Trường hợp nếu không hỗ trợ sử dụng kiểu HalfTensor 
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # Định nghĩa mô hình kiến trúc Transformer 
        model = Transformer(model_args)
        # Tải trạng thái mô hình từ một Checkpoint sử duing strict = False để bỏ qua 
        # các khóa không khớp mà không gây ra lỗi 
        model.load_state_dict(checkpoint, strict=False)
        # Tính và in ra thời gian cần thiết để tải mô hình
        print(f"Load in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)
    

    # Thiết lập phương thức khởi tạo và định nghĩa các thuộc tính 
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        # self.model định nghĩa mô hình 
        self.model = model 
        self.tokenizer = tokenizer 
        # ChatForm 
        self.formatter = ChatFormat(tokenizer)

    
    # @torch.inference_mode() là một decorator được sử dụng để tối ưu hóa hiệu suất khi mô hình đang ở chế độ suy luận (inference). 
    # Khi sử dụng decorator này, PyTorch sẽ tắt việc theo dõi các thay đổi trên tensor để cải thiện hiệu suất, vì không cần thiết phải
    # Tính toán gradient 
    @torch.inference_mode()
    def generate(
        self,
        # prompt_tokiens một dnah sách các token _ids đầu vào dùng để khởi đầu quá trình 
        # sinh. Mỗi dnah sách con trong nó chứa các token của mọt chuỗi đầu vào riêng lẻ 
        prompt_tokens: List[List[int]],
        # Max_gen_len là độ dài tối đa của chuỗi mà mô hình sẽ sinh ra. Nó giới hạn số lượng token 
        # tối đa mà mô hình có thể tạo ra trong mỗi chuỗi 
        max_gen_len: int,
        # temperature: float = 0.6: Tham số này điều chỉnh độ ngẫu nhiên trong quá trình sinh. 
        # Một giá trị thấp hơn làm cho kết quả ít ngẫu nhiên hơn, trong khi giá trị cao hơn làm cho kết quả đa dạng
        temperature: float = 0.6,
        # Top_p xác suất top p để lấy mẫu hạt nhân được định nghĩa là một ngưỡng gía trị
        top_p: float = 0.9,
        # Xác xuất ghi nhật ký chỉ định có ghi nhật ký hay không 
        logprobs: bool = False,
        # cờ cho biết có bao gồm mã thông báo nhắc nhở trong kết quả được tạo hay khônng 
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        # Lấy ra cấu hình từ điển tham số của mô hình 
        params = self.model.params 
        # lấy ra số lượng token trong chuỗi lời nhắc đầu vào 
        bsz = len(prompt_tokens)
        # Đảm bảo rằng số lượng token của đầu vào không vượt quá số lượng tokens mà mỗi batch có 
        # thể chứa đựng 
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # min_prompt_eln lưu trữ độ dài ngắn nhất  của một chuỗi token trong danh sách 
        min_prompt_len = min(len(t) for t in prompt_tokens)
        # tương tự như min thì max ngược lại sẽ lưu trữ độ dài cuatr chuỗi lớn nhất 
        max_prompt_len = max(len(t) for t in prompt_tokens)
        # Đảm bảo rằng độ dài tối đa của chuỗi lời nhắc không vượt quá độ dài tối 
        # đa cho phép max_seq_len 
        assert max_prompt_len <= params.max_seq_len
        # Total_len được đặt thành giá trị nhỏ nhất giữa params.max_seq_len và max_gen_len + max_prompt_len (tổng độ dài của chuỗi đầu ra và chuỗi prompt).
        # Biến anyf được sử dụng để định kích thước tối đa cho chuỗi đầu vào và đầu ra trong 
        # quá trình xử lý 
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # sử dụng tokenizer để gọi đến thuộc tính pad_id kết quả gán cho pad_id 
        # giá trị này được sử dụng để đẹm văn bản 
        pad_id = self.tokenizer.pad_id
        # Tạo một tensor mới có kích thước bsz * total_len giá trị mặc định được đặt bằng pad_id 
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        # duyệt qua danh sách các token đầu vào lấy ra các ids và id của chúng 
        for k, t in enumerate(prompt_tokens):
            #  gán các token trong chuỗi prompt hiện tại (t) vào một phần của tensor tokens.
            # thêm k và t tương ứng vào một phần của tensor token, k sẽ đựoc sử dụng để chọn batch tương ứng 
            # với chuỗi prompt hiện tại . Các token trong chuỗi đầu vào được chuyển thành ténor 
            # với kiểu torch.long  
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        # kiểm tra biến logprobs.
        if logprobs:
            # sao chép tensor tokens với các giá trị lấp đầy = 0
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        # khởi tạo biến prev_pos gán giá trị = 0 biến này thể hiện vị trí position của token 
        # trước đó 
        prev_pos = 0
        # khởi tạo 1 tensor shape = bsz các giá trị được lấp đầy = False 
        eos_reached = torch.tensor([False] * bsz , device='cuda')
        # kiểm tra xem chỉ số của token có tương ứng với pad_id không nếu nó tương ứng
        # thì giá trị trong input_text_mask = True và ngược lại sẽ có kết quả = False 
        input_text_mask = tokens != pad_id
        # Kiểm tra xem độ dài tối thiểu min_prompt_len có bằng với total_len 
        if min_prompt_len == total_len: 
            # Nếu như điều này đúng thực hiện  phương thức forward (lan chuyền tiến)
            # để có thể thực hiện dự đoán token tiếp theo dựa vào token trước đó 
            logits = self.model.forward(tokens, prev_pos)
            # compute loss function 
            token_logprobs = F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                # reduction = none yêu cầu entropy chéo trả về mất mát của từng token riêng lẻ 
                # không phải một giá trị mất mát tổng thể duy nhất
                reduction="none",
                # Ignore_index phớt lờ đi các pad_id bỏ qua việc tính toán các token này
                ignore_index=pad_id,
            )
        
        # Gọi phương thức self.tokenizer.stop_token để lấy ra các giá trị token kết thúc văn bản và kết thúc lượt trong 
        # hội thoại tin nhắn văn bản sau đó hàm list được sử dụng chuyển đổi các giá trị này thành các danh sách 
        # cuối cùng hàm torh.tensor được gọi để chuyển đổi danh sách này thành tensor 
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))
        # sử dụng vòng lặp for lặp từ min_prompt_len đến total_len điều này có nghĩa là vòng lặp sẽ chạy 
        # đến khi đạt được độ dai dàimong muốn của văn bản 
        for cur_pos in range(min_prompt_len, total_len):
            # cắt một đoạn biểu diễn [prev_pos:cur_pos] tức là một số lượng các token được cắt 
            # và sử dụng token trước đó của đoạn này để dự đóan token tiếp theo 
            # việc này sử dụng một đoạn văn bản để dự đoán một token tiếp theo 
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # kiểm tra xem xác suất sinh token ngẫu nhiên nếu như xác suất này > 0
            if temperature >  0: # xác suất này chỉ định nếu như token sinh ra lớn hơn giá trị này sẽ được trọn 
                # thực hiện chia các xác suất logits cho temperature và áp dụng hàm softmax 
                # lên kết quả này. Điều này có tác dụng làm phẳng phân bố xác suất tăng khả năng chọn 
                # cho các token ít khả năng hơn 
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # sau đó thực hiện lấy mẫu top_p cho các tokens
                next_token = sample_top_p(probs, top_p)
            # Trường hợp xác suất temperature không tồn tại 
            else:
                # Lấy trực tiếp token có xác suất cao nhất 
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            # sau đó giá trị token được lấy next_token sẽ được chuyển đổi thành tensor 1 chiều 
            next_token = next_token.reshape(-1)
            # Thay thế giá trị của token trong danh sách input_text_mask và tokens tại chỉ số 
            # cur_pos thành tokens được chọn là token tiếp theo 
            next_token = torch.where(
                # Tensor_text_mask lưu trữ thông tin về các token nguyên mẫu 
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            # sau đó cập nhật token 
            tokens[:, cur_pos] = next_token
            # kiểm tra xem có cho phép ghi nhật ký logits không 
            if logprobs: 
                # Thực hiện tính toán Loss cho các token vừa được dự đoán 
                # sử dụng lát cắt prev_pos + 1 : cur_pos + 1: Chỉ định phần tử từ prev_pos + 1 đến cur_pos 
                # (không bao gồm cur_pos + 1) trong trục thứ hai của token_logprobs
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = - F.cross_entropy(
                    # Tính toán entropy chéo ấm giữa các logits được chuyển trước 
                    inputs = logits.transpose(1,2),
                    target = tokens[:, prev_pos + 1 : cur_pos + 1], 
                    # reduction = none được sử dụng để tính toán độ mất mát cho từng token riêng lẻ 
                    reduction = "none", 
                    ignore_index=pad_id, # bỏ qua việc tính toán các token padding
                )
            
            # Cập nhật biến eos_reached 
            # kiểm tra xem kết token tiếp theo có phải là dấu hiệu kế thúc chuỗi và không phải 
            # là một phần của văn bản đầu vào hay không. Nếu cả 2 điều kiện đúng nó sẽ đánh dấu rằng 
            # chuỗi đã kết thúc 
            # Toán tử |= được sử dụng để cập nhật biến eos_reached thông qua một phép so sánh 
            # toán tử ~ là phép phủ định được áp dụng cho tensor boolean input_text_mask
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                # hàm torch isin kiểm tra xem next_tokens có nằm trong danh sách stop_tokens 
                torch.isin(next_token, stop_tokens)
            )

            # cập nhật vị trí trước đó thành vị trí hiện tại để bắt đầu 
            # cho dự đoán token tiếp theo 
            prev_pos = cur_pos
            # Kiểm tra xem tất cả các điều kiện trong eos đều đúng hay không 
            if all(eos_reached):
                # Nếu tất cả đều đúng thì vòng lặp sẽ kết thúc 
                break 
        

        # Kiểm tra xem có cho phép ghi lại xác suất logit hay không 
        if logprobs: 
            # Nếu có chuyển tensor chứa các kết quả xác suất logit của token 
            # thành một danh sách 
            token_logprobs = token_logprobs.tolist()
        
        # Khởi tạo 2danh sách là out_token chứa các token đầu ra 
        # và out_logprob sẽ chứa các xác suất của các token tương ứng
        out_tokens, out_logprobs = [], []
        # duyệt qua các phần tử trong danh sách tolist
        # Lấy ra giá trị và chỉ số của từng token 
        for i, toks in enumerate(tokens.tolist()):
            # cắt theo độ dài tối đa của dnah sách token đầu vào 
            # khởi tạo biến start gán nó = 0 nếu như echo = True nếu không thì gán nó 
            # bằng độ dài của danh sách prompt_token theo chỉ số i 
            start = 0 if echo else len(prompt_tokens[i])
            # Thực hiện cắt token từ chỉ số start đến max_gen_len  độ dài tái tạo tối đa 
            # điều này đảm bảo các token kết quả nằm trong giới hạn độ dài được chỉ định 
            toks = toks[start : len(prompt_tokens[i] + max_gen_len)]

            # Gán xác suất probs = None 
            # probs để lưu trữ các xác suất tương ứng cho token (nếu có).
            probs = None 
            # kiểm tra xem biến logsprobs có được thiết lập 
            if logprobs:
                # khởi tạo danh sách logprobs danh sách này sẽ chứa các giá trị xác suất của token hiện tại 
                # được biểu diễn bởi các xác suất start : len(prompt_tokens[i]) + max_gen_len 
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            
            # cut to after eos tok if any
            # Kiểm tra token kết thúc token này là một chuỗi nằm trong danh sách stop_tokens
            for stop_token in self.tokenizer.stop_tokens:
                # Xử lý một ngoại lệ ngoại lệ này cố gằng tìm token kết thúc eos
                try:
                    # Nếu tìm thấy stop_token trong dah sách token hàm indx sử dụng
                    # để đi lần lượt các chỉ số  
                    eos_idx = toks.index(stop_token)
                    # Thực hiện một lát cắt cho danh sách tokens từ 0-> eos_idx token 
                    toks = toks[:eos_idx]
                    # tương tự thực hiện cắt các xác suất trong tensor probabilities 
                    # để bỏ qua token sau token đánh giấu nhưng với điều kiện xác suất log được thiết lập 
                    probs = probs[:eos_idx] if logprobs else None
                # Nêys như Khối ngoại lệ trên lỗi 
                # hay không tìm thấy token eos thì ném ra lối ValueError 
                except ValueError:
                    pass
            # Thêm các token vào danh sách out_token
            out_tokens.append(toks)
            # và các logits vào danh sách out_logits 
            out_logprobs.append(probs)
        # Cuối cùng trả về danh sách token và xác suất đầu ra tương ứng của chúng 
        # với điều kiện xác suất logits được thiết lập 
        return (out_tokens, out_logprobs if logprobs else None)#
    

    # Thiết lập phương thức text_completion có chức năng hoàn thành biểu diễn 
    # văn bản cho một danh sách các lời nhắc sử dụng mô hình ngôn ngữ sinh tạo 
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]: 
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        # Kiểm tra xem tham số max_gen_len có được thiết lập hay không 
        if max_gen_len is None: 
            # Nếu tham số này không được thiết lập 
            # truy suất tham số này từ từ điển tham số của mô hình 
            max_gen_len = self.model.params.max_seq_len - 1
        
        # Hàm self.tokenizer.encode được gọi để giải mã các mã thông báo text trong lời 
        # nhắc văn bản đầu vào 
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # Thực hiện sinh văn bản cho lời nhắc hàm generate được gọi để dự đoán token tiếp theo #
        # từ danh sách token đầu vào kết quả trả về danh sách token được tạo và danh sách xác suất của chúng 
        generation_tokens, generation_logprobs = self.generate(
            # Truyền đầu vào 
            prompt_tokens=prompt_tokens, 
            # kích thước tối đa của chuỗi được tạo ra
            max_gen_len=max_gen_len, 
            # Tham số lấy xác suất token > 0.6 
            temperature = temperature, 
            # Mẫu top_p 
            top_p=top_p, 
            # Xác suất nhật ký logit 
            logprobs=logprobs,
            # Biến chỉ định boolean echo cho biết mã thông báo kết quả có được 
            # tạo ra hay không
            echo=echo,
        )
        # kiểm tra xem logbprobs có được thiết lập hay không 
        if logprobs: 
            # Nếu tham số này được thiết lập trả về một danh sách 
            # danh sách này sẽ là một từ điển chứa các keys 
            return [
                {
                    # Generation đoạn văn bản chứa danh sách các token được tạo ra 
                    "generation": self.tokenizer.decode(t),
                    # và danh sách các token của từng đoạn văn bản t 
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    # logbprobs xác suất của các tokens được tạo ra 
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        # Nếu không chỉ trả về danh sách của các đoạn văn bản được tạo ra 
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    # Xây dựng phương thức chat_completion phương thức này sẽ thực hiện việc 
    # thực hiện hoàn đoạn chat văn bản 
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        # kiểm tra xem max_gen_len có được thiết lập hay không
        if max_gen_len is None:
            # Nếu không ta kêu gọi đến tham số max_seq_len từ mô hình 
            max_gen_len = self.model.params.max_seq_len - 1
        
        # Giải mã các đoạn tin nhắn văn bản có trong đoạn hội thoại 
        # hàm self.formatter.encode_dialog_prompt được gọi để giải mã lần lượt các đoạn hội thoại kết quả là một danh sách 
        # chứa các tensor tokens 
        prompt_tokens  = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs 
        ]
        # sau đó thực hiện sinh văn bản dựa vaò đoạn hội thoại văn bản 
        generation_tokens, generation_logprobs = self.generate(
            # danh sách đầu vào 
            prompt_tokens=prompt_tokens,
            # kích thước tối đa của chuỗi được tạo ra
            max_gen_len=max_gen_len, 
            # Tham số lấy xác suất token > 0.6 
            temperature = temperature, 
            # Mẫu top_p 
            top_p=top_p, 
            # Xác suất nhật ký logit 
            logprobs=logprobs,
            
        )  

        # kiểm tra xem biến logbprobs có được thiết lập hay không
        if logprobs : 
            # Nếu nó được thiết lập trả về một danh sách 
            return [
                # danh sách này sẽ chứa một từ điển 
                {
                    # key generation có values là một từ điển từ điển này sẽ chứa nội dung và 
                    # nguồn từ nội dung 
                    "generation":{
                        # role thể hiện vai trò của đoạn hội thoại 
                        "role": "assistant",
                        "content": self.tokenizer.decode(t), # đây là nội dung của đoạn hội thoại
                    },
                    # token danh sách chứa các token của mỗi đoạn nội dung t 
                    "token": [self.tokenizer.decode([x]) for x in t],
                    # logprobs chứa xác suất của các token theo từng đoạn nôi dung t
                    "logprobs": logprobs_i,
                }
                # duyệt qua dnah sách generation_tokens và generation_logprob để lấy 
                # ra danh sách các token được sinh ra và xác suất của chúng 
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        
        # Trường hợp còn lại trả về một danh sách chứa một từ điển 
        # với key là generation đây là một từ điển chứa đoạn hội thoại văn bản được đưa vào
        return [
                {
                "generation": {
                    # từ điển này chỉ định đoạn hội thoại của người trợ lý 
                    "role": "assistant",
                    # và nội dung của được cung cấp
                    "content": self.tokenizer.decode(t),
                },
            }
            # duyệt qua các đoạn hội thoại t đã được tạo ra bởi mô hình
            for t in generation_tokens
        ]
    

# Thiết lập phương thức lấy mẫu top_p 
# Phương thức này lấy mẫu tập hợp mã thônng báo nhỏ nhất có xác suất vượt 
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    # sử dụng hàm torch.sort để sắp xếp các giá trị tỏng tensor xác suất
    # hàm này sẽ trả về danh sách được sắp xếp và chỉ số của từng xác suất 
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # sử dụng hàm torch.cumsum để tính tổng tích lũy các giá trị trong tensor probs_sort 
    # Ví dụ a = torch.tensor([1, 2, 3, 4])
    # tổng tích lũy của nó là torch.cumsum(a, dim=0) tensor([1, 3, 6, 10])
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # sau đó kiểm tra xem hiệu kết quả tích lũy này với giá trị xác suất p
    mask = probs_sum - probs_sort > p
    # các giá trị mask (False) sẽ được gán = 0.0
    probs_sort[mask] = 0.0
    # sử dụng hàm div_ để thực phép chia trên tensor  gốc mà không tạo ra bản sao 
    # Mục đích chuẩn hóa probs_sort sao cho tổng xác suất trên mỗi hàng bằng 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # torch.multinomial: Hàm lựa chọn ngẫu nhiên một số lượng mẫu từ một phân phối xác suất.
    # num_samples=1: Số lượng mẫu cần lấy, ở đây là 1.
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # torch.gather: Hàm trích xuất các giá trị từ tensor dựa trên chỉ số cụ thể.
    # thực hiện trích xuất tensor probs_ids tại chiều cuối cùng theo 
    # chỉ số next_token 
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token