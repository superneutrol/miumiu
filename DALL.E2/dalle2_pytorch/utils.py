import time 
import importlib 

# Xây dựng các hàm chức năng 
def exists(val): 
    # trả về chính nó nếu như nó có tồn tại 
    return val is not None 

# Hàm trợ giúp tính toán thời gian 
class Timer: 
    # khởi tạo 
    def __init__(self):
        # đặt lại dấu thời gian hiện tại 
        self.reset()

    # Thiết lập hàm reset để đánh dấu một mốc thời gian đã qua 
    def reset(self):
        self.last_time = time.time()
    
    # Tính toán thời gian đã qua bằng cách lấy thời gian hiện tại 
    # - mốc thời gian được đánh dấu khi bắt đầu 
    def elapsed(self):
        return time.time() - self.last_time
    

# Định nghĩa một hàm prin_ribon trả về một chuỗi với s được bao quanh bởi 1 dải ký tự sumpol
# print helpers

def print_ribbon(s, symbol = '=', repeat = 40):
    flank = symbol * repeat
    return f'{flank} {s} {flank}'

# import helpers

def import_or_print_error(pkg_name, err_str = None):
    try:
        # thực hiện import pkg_name 
        return importlib.import_module(pkg_name)
    # nếu như import gặp lỗi  moduleNotFoundError 
    except ModuleNotFoundError as e:
        # hàm này sẽ in ra chuỗi lỗi err_str 
        if exists(err_str):
            print(err_str)
        exit() 