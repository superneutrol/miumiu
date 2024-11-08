"""Xây dựng các hàm tiện ích tương tác với khung giới hạn và GIou. 
    Utilities for bounding box manipulation and GIou.
"""
import torch 
from torchvision.ops.boxes import box_area 

# Thiết lập phuuwong thức xử lý nhận đầu vào là danh sách các hộp toạ 
# trung tâm - chiều rộng , chiều cao  sang định dạng tọa độ góc 

# Vì để tìm được khoảng cách từ tâm đến chiều cao và chiều rộng thì ta * 1/2 h , w
# hộp giới hạn là một hình chữ nhật thì khoảng cách từ tâm đến chiều dài = 1/2 chiều rộng 
# và khoảng cách từ tâm đến chiều rộng = 1/2 chiều dài để thực hiện phép tính khoảng cách 
# trong không gian nhiều chiều sẽ là tọa độ a - b 
def box_cxcywh_to_xyxy(x):
    # sử dụng hàm unbind để tách tensor x thành các các tensor con 
    # bằng cách loại bỏ 1 chiều cụ thể ở đây tách theo chiều cuối cùng 
    # tức là chiều thứ 2 [N, 4] chứa các giá trị trọng tâm và w h của hộp 
    # giới hạn  x_c và x_y là tọa độ của trọng tâm theo trục x và y 
    x_c, y_c, w, h = x.unbind(-1)
    # tính toán các khoảng cách từ trọng tâm đến các cạnh của hộp giới hạn 
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), # Tạo độ x y dưới phải 
         (x_c + 0.5 * w), (y_c + 0.5 * h)] # Tọa độ x y trên trái 
    # sau đó nối các giá trị nàu lại với nhau dưới dạng stack 
    return torch.stack(b, dim=-1)


# Thiết lập phương thức từ hộp tọa độ xy , xy lấy ra trọng điểm 
def box_xyxy_to_cxcywh(x):
    # sử dụng hàm unbind để tách tensor x thành các tensor con 
    # bằng cách loại bỏ 1 chiều cụ thể đây là chiều cuối trong tensor x
    # (x0, y0) là tọa độ trên trái còn (x1, y1) là tọa độ dưới phải 
    x0, y0, x1, y1 = x.unbind(-1)
    # Tính toán giá trị của a điểm 0 , 1  và b 1 , 2 / 2 để tính được khoảng 
    # cách tâm đến các cạnh của hôp giới hạn (x0 + x1) = độ dài cạnh / 2 = khoảng cách từ chiều dọc đến tâm 
    b = [(x0 + x1) / 2, (y0 + y1) / 2, # tọa độ trên trái  (2, 3) (4, 1) = (c x, y) = (2, 2)
         # sau đó tính toán phép trừ tọa độ không gian để tính độ dài w và h
         (x1 - x0), (y1 - y0)] 
    # nối dnah sách các giá trị này lại với nhau dưới dạng stack
    return torch.stack(b, dim=-1)


# Thiết lập phương thức tính toán độ chồng lấn iou cho hộp giới hạn 
def box_iou(boxes1, boxes2):
    # lấy ra diện tich của 2 hộp giới hạn 
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # Tính toán tọa độ gia điểm của hộp giới hạn 
    # top_left (tọa độ trên bên trái) sử dụng max để hộp không lệch lên trên 
    # mảng boxes1  shape = [N , 4 ] chưa tọa độ hộp giới hạn N là số lượng phần 
    # tử và mỗi phần tử có 4 tọa độ . Với boxes[:, None :2] có nghĩa là lấy 2 phần tử đầu 
    # tiên của 1 Tensor shape [N, 1, 4] ký hiệu [:,] nghĩa là lấy tất cả các giá trị theo 
    # 1 chiều cụ thể , [None], đặt ở vị trí nào thì có nghĩa thêm 1 chiều vào vị trí đấy 
    # và :2 nghĩa là lấy đi 2 phần tử đầu của danh sách .
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # sử dụng max để xác định các khi vực giao nhau của hộp giới hạn để lấy ra 
    # tọa độ có giá trị lớn nhất phía trên bên trái và phía dưới bên phải.
    # còn tính toán tọa độ bottom_right = với 2: có nghĩa là lấy 2 phần tử cuối cùng dnah sách
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    # tính chiều rộng và chiều cao của phần giao nhau giữa các hộp giới hạn
    # phép trừ rb - lt là kích thước của phần giao nhau nhưng có thể là âm do đó '
    # sử dụng clamp(min=0) đặt giá trị tối thiểu = 0
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # Tính diện tích của phần giao nhau bằng cách nhân chiều rộng 
    # (wh[:, :, 0]) và chiều cao (wh[:, :, 1]) của nó.
    # tensor mh là tensor mà chiều cuối là chiều mang giá trị biểu diễn  = [2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    # tính diện tích của phần hợp nhất của 2 hộp giới hạn bằng cách cộng tổng 
    # diện tích trừ đi phần mà 2 diện tích dao nhau 
    union = area1[:, None] + area2 - inter
    #  Tính tỷ lệ IOU bằng cách chia diện tích giao nhau inter cho diện tích hợp nhất union.
    iou = inter / union
    # trả về tỷ lệ iou và union (diện tích giao)
    return iou, union

# thiết lập phương thức tổng quát box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # kiểm tra xem 2 giá trị top_left có lớn hơn right_bottom trong 2 hộp giới hạn hay không
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    # Tính toán độ chồng lấn và diện tin=chs giao nhau của 2 hộp giới hạn
    iou, union = box_iou(boxes1, boxes2)

    # Tính toán tọa độ top_left nhỏ nhất giữa 2 hộp giới hạn thay vì     
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    # Tính chiều dài và chiều rộng hộp giới hạn 
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # tính toán diện tích hộp tọa độ 
    area = wh[:, :, 0] * wh[:, :, 1]
    # trả về kết quả độ trồng lấn trừ hiệu diện tích hộp - diện tích giao chia cho diện tích 
    # sử dụng để điều chỉnh giá trị iou dựa trên tỷ lệ của phần giao nhau so với 
    # tổng diện tích. Đây là một cách để cải thiện độ chính xác của phép đô IOU.
    return iou - (area - union) / area

# xây dựng mătj nạ cho hộp tọa độ 
# có chức năng để lấy các tọa độ cho hộp giới hạn từ mặt nạ 
def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    # kiểm tra xem số phần tử của mask == 0
    if masks.numel() == 0:
        # nếu không có trả về 1 tensor shape = [0,4]
        return torch.zeros((0, 4), device=masks.device)
    # lấy ra 2 giá trị cuối cùng của máks là h và w shape [N, h , w]
    h, w = masks.shape[-2:]
    # tạo ra 2 mảng x và i chứa chỉ số tương ứng viwis chiều rộng và chiều cao 
    # 1: Tạo ra các tensor chứa chỉ số tọa độ cho mỗi điểm trên mặt nạ
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    # xử dụng x , y để tạo ra 1 lưới tọa độ 2D
    y, x = torch.meshgrid(y, x)

    # [N , H , W] *[1, W] -> Shape = [N, H, W] * phép nhân từng mặt nạ với lưới tọa độ 
    x_mask = (masks * x.unsqueeze(0))
    # làm mịn y_mask theo chiều thứ 2 tức H và tham số -1 là lấy giá trị lớn nhất 
    # theo chiều cuối cùng , hàm max sẽ trả về 1 tuple gồm chỉ số và giá trị [0]
    # để lấy giá trị của tuple và bỏ qua chỉ số
    x_max = x_mask.flatten(1).max(-1)[0] # Flatten chuyển từ tensot [N , H , W] Thành N , [W*H]
    # Thực hiện hàm masked_fill để lấp đầy tensor mask tensor mask là tensor có giá trị bool
    # với toán tử ~ có chức năng đổi ngược các giá trị mask và với mask_fill thay thế các giá 
    # trị năm ngoài mặt nạ  = 1e8 .  Sau đó chuyển đổi hình dạng từ tensor 2 chiều 
    # thành 1 chiều và tìm ra giá trị nhỏ nhất theo chiều cuối cùng và lấy giá trị của nó 
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    # [N , H , W] *[1, H] -> Shape = [N, H, W] Ở đây không thực hiện phép nhân thông thường
    # phép nhân element-wise (tức là nhân từng phần tử tương ứng với nhau) thông qua broadcasting
    # tensor y sẽ được mở rộng có kích thước phù hợp với tensor mask
    y_mask = (masks * y.unsqueeze(0))
    # làm phẳng tensor này và lấy ra giá trị lớn nhất theo chiều cuối
    # là tọa độ y max
    y_max = y_mask.flatten(1).max(-1)[0]
    # tương tự như trên đảo ngược các giá trị trong mask và lấp đầy các gái trị 1e8 cho các 
    # giá trị nằm ngoài mặt nạ . tiếp tục đưa nó thành tensor 1 chiều và lấy giá trị nhỏ nhất
    # là tọa độ  y min 
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    # Kết hợp các tọa độ thành tensor kết quả
    # chỉ định các tensor được nối lại theo chiều thứ 2 
    return torch.stack([x_min, y_min, x_max, y_max], 1)