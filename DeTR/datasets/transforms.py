"""Máy biến áp và tăng cường hình ảnh cho toàn bộ ảnh + bounding box"""

import random 
import PIL 
import torch 
import torchvision.transforms as T 
import torchvision.transforms.functional as F 
import Utils
from Utils.box_ops import box_xyxy_to_cxcywh # hàm chức năng tách tọa độ hộp giới 
# hạn x, x1 , y, y1 thành tọa độ trọng tâm x , y ,với kíc thước 2 chiều h , w
from Utils.misc import interpolate # hàm nội suy interpolate gồm tham số fract_scale 
# một vài tham số khác để quyết định cho việc lấy mẫu hay giảm mẫu . 

# Thiết lập phương thức crop để tính toán và thực hiện căn chỉnh hộp giới hạn cho hình ảnh 
# phương thức này nhận đầu vào gồm  hình ảnh , hộp giới hạn thực của hình ảnh , 
# vùng đề suất của hình ảnh 
def crop(image, target, region):
    #F.crop là một hàm được sử dụng để cắt hình ảnh. 
    # target là một từ điển chứa thông tin liên quan đến hình ảnh như 
    # hộp giới hạn, nhãn, v.v. region là một tuple chứa thông tin 
    # vùng cắt (i, j, h, w) tương ứng với tọa độ x, y và chiều cao,
    # chiều rộng của vùng cắt.
    cropped_image = F.crop(image, *region)
    # sao chép thông tin target để không thay đổi làm mất mát thông tin gốc 
    target = target.copy()
    # lấy thông tin vùng cắt : tọa độ trên trái x, y và chiều cao , chiều rộng 
    i, j , h , w = region 
    # cập nhật kích thước mới h và w của hộp giới hạn sau khi cắt  vào target 
    target['size'] = torch.tensor([h, w])
    # tạo một dnah sách các trường thông tin cần cập nhật 
    fields = ['labels', 'area', 'iscrowd']

    # nêu có thông tin về hộp giới hạn trong danh sách target 
    if "boxes" in target:
        # gán cho target = thông tin của boxes trong danh sách target 
        # boxes shape [ x_min, y_min , x_max, y_max] là các giá trị tọa độ
        boxes = target['boxes']
        # tạo môt tensor max_size shape = [w*h] là kích thước tối đa mà hộp 
        # giơi hạt có thể đật được 
        max_size = torch.as_tensor([w,h], dtype=torch.float32)
        # tính toán hộp giới hạn mới sau khi cắt  
        # tọa độ mới của hộp giới hạn trên hình ảnh đã cắt sẽ là (x_min - j, y_min - i, x_max - j, y_max - i)
        # tương tự như 1 phép căn chỉnh phù hợp cho mỗi lát cắt 
        cropped_boxes = boxes - torch.as_tensor([j, i , j , i])
        # giới hạn hộp giới hạn mới để cho nó có kích thước không vượt quá 
        # kích thước tối đa . Tham số (-1, 2, 2) được hiểu là 
        # có N hộp mỗi hộp có 4 giá trị thể hiện tọa độ 2 điểm sẽ được biểu diễn bằng ma trận 2 * 2
        # x_max là tham số sẽ được điều chỉnh tùy thuộc vào cropped_boxes
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2 , 2), max_size)
         # Đảm bảo hộp giới hạn mới không có giá trị âm
        cropped_boxes = cropped_boxes.clamp(min=0)
        # Tính diện tích của hộp giới hạn mới copped_boxes shape =
        # [n , 2 , 2] khi cắt 0 , 1 nghĩa là chiều thứ 2 sẽ tách thành 2 cột riêng 
        # và cho giá trị trên trái và dưới phải của hộp  
        # phép trừ tọa độ này x -> w và y > h tức là phép trừ khoảng cách theo trục
        # sử dụng prod để tính tích trên chiều thứ 2 của 2 tensor shape = [n, 2(h, w của phép trừ), 2]
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        # Cập nhật hộp giới hạn và diện tích vào target
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        # gán cho diện tích trong dnah sách target = diện tích mới 
        target["area"] = area
        # Thêm "boxes" vào danh sách trường thông tin cần cập nhật
        fields.append("boxes")

    # kiểm tra xem danh scahs target có tồn tại mặt nạ 
    if "masks" in target: 
        # áp dụng lên target một mặt nạ 
        # mặt nạ này được cắt theo kích thước theo chỉ số hàng i đến i + h và cột j đến j+w
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks") # thêm chuỗi "masks" vào danh sách fields, 
        # có thể được sử dụng để theo dõi các trường dữ liệu nào đã được cập nhật hoặc xử lý.

    # thực hiện việc xáo đi accs phần tử khi hộp giới hạn hoặc masks có điện  tích  = 0
    if "boxes" in target or "masks" in target:
        # thực hiện việc lựa chọn hộp ưu tiên khi xác đinh phần 
        # tử nào được giữ lại (keep)
        # điều này tương thíc(compatible) với việc thực hiẹne trước đó 
        if "boxes" in target:
            # định hình các họp giới hạn trong danh sách target 
            cropped_boxes = target["boxes"].reshape(-1, 2 , 2)
            # thực hiện so sánh x1, x0 là tọa độ dưới và trên nếu như tọa độ dưới phải 
            # mà có giá trị x nhỏ hơn giá trị x0 của tọa độ trên trái 
            # với torch.all (dim=1) tức là áp dụng phép so sánh này cho chiều thứ 2
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        # trường hợp còn lại 
        else: 
            # làm phẳng các mặt nạ mask theo chiều thứ 2 shape [N, @*@]
            # với chiều thứ 2 chứa tất cả thông tin biểu diễn cho mỗi mặt nạ 
            # .any(1) kiểm tra xem mỗi hàng trong tensor 2D mask nếu có bất kỳ gía trị
            # nào  = 0 . Nếu có ít nhất một giá trị khác 0 trong hàng, nó sẽ trả về True cho hàng đó.
            keep = target['masks'].flatten(1).any(1)
        # duyệ qua các chuỗi trong danh sách fields 
        for field in fields:
            # ánh xạ các chuỗi field trong danh sách target cho các
            # giá trị theo chuỗi filed được áp dụng keep để nhận được 
            # 1 danh sách target với field đã được thực hiện loại bỏ 
            target[field] = target[field][keep]
    # trả về tập hợp các hình ảnh đã được cắt + danh sách target 
    return cropped_image, target

# Thiết lập phương thức hflip sử dụng để lật hình ảnh 
# nhận đầu vào là danh sách như một kiểu từ điển gồm một chuỗi và theo sau nhiều giá trị 
def hflip(image, target):
    # sử dụng lật hình ảnh theo phép lật honrizontal
    flipped_image = F.hflip(image)
    # trích xuất chiều cao và chiều ngang của hình ảnh 
    w, h = image.size 
    # sao chép thư mục target để tránh việc các giá trị trong target bị thay đổi 
    target = target.copy()
    # kiểm tra xem giá trị boxes có tồn tại trong target ;
    if "boxes" in target:
        # gán cho biến boxes bằng giá trị của boxes trong danh sách target
        boxes = target["boxes"]
        # hóa vị các giá trị của tensor boxes theo chiều thứ 2
        # [x_min, y_min, x_max, y_max], hoán vị này sẽ thay đổi nó thành [x_max, y_min, x_min, y_max].
        # Sau đó thực hiện phép nhân với tensor [-1, 1, -1, 1 ] để đảo ngược giá trị
        # cuối cùng cộng các giá trị này với [w, 0, w, 0] cho tọa độ x_min, x_max đã 
        # được đảo ngược. Điều này đưa các tọa độ x về đúng vị trí sau khi lật ảnh  
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        # gán cho giá trị boxes tromng danh sách target = các boxes sau khi lật 
        # mục đích là giúp thuẹc hiện việc tăng cường hình ảnh cho mô hình
        target["boxes"] = boxes
    # kiểm tra xem trong danh sách target có tồn tại giá trị mask 
    if "masks" in target:
        # lật mặt nạ mask theo chiều ngang tương tự như việc lật các hình ảnh theo chiều ngang
        target['masks'] = target['masks'].flip(-1)

    # trả về các hình ảnh đã được lật , danh sách chứa mặt nạ lật , và hộp giới hạn lật 
    return flipped_image, target


# Thiết lập phuuwong thức resize có chức năng thay đổi kích thước hình ảnh nhỏ 
def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    # điều chỉnh kích thước hình ảnh mà vẫn giữ được tỷ lệ khung hình (aspect ratio)
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        # lấy chiều rộng và chiều cao hiện tại của hình ảnh 
        w, h = image_size
        # kiểm tra xem có giới hạn kích thước tối đa không 
        if max_size is not None:
            # tìm một kích thước nhỏ nhất giữ chiều rộng và chiều cao
            # sau đó chuyển nó thành số thực 
            min_original_size = float(min((w, h)))
            # tương tự như trên tìm kích thước lớn nhất giữa chiều rộng và chiều cao 
            # và chuyển nó thành số thực
            max_original_size = float(max((w, h)))
            # kiểm tra xem kích thước nhỏ có vượt qua giới hạn tối đa hay không 
            # tức là tỷ lệ khung hình tính toán có > max_size không tức là có vượt quá sự cho phép 
            if max_original_size / min_original_size * size > max_size:
                # nếu có thì tính toán lại kích thước mới sao cho nó không vượt quá giới hạn 
                #  Nếu điều kiện trên đúng, kích thước mới sẽ được tính toán lại để không vượt quá max_size. 
                size = int(round(max_size * min_original_size / max_original_size))

        # kiểm tra xem w< h anh w == size .. 
        if (w <= h and w == size) or (h <= w and h == size):
            # nếu có trả về h và w của hình ảnh 
            return (h, w)    
        # nếu w < h 
        if w < h : 
            # nếu đúng gán ow = size
            ow = size 
            # và oh bằng được tính toán sao cho tỷ lệ khung hình giữ được nguyên 
            oh = int(size * h / w)
        # trường hợp còn lại 
        else:
            # gán cho oh = size 
            oh = size
            # ow  = size * w / h  và chiều rộng mới ow sẽ được tính toán tương tự.
            ow = int(size * w / h)
        # tỷ lệ khung hình là tỷ lệ được tính bởi chiều rộng và chiều cao của hình ảnh 
        # cuối cùng trả về oh và ow là 2 giá trị lưu trữ kích thước của hình ảnh h, w sau khi thay đổi 
        return (oh , ow)
    
    # Thiết lập phương thức get_size để nhận kích thước mới của 1 hình ảnh dựa trên 
    # tham số truyền vào 
    def get_size(image_size, size, max_size=None):
        # kiểm tra xem size có phải là list hay tuple
        if isinstance(size, (list, tuple)):
            # nếu đúng trả về size với kích thước mới 
            # Trả về một tuple với thứ tự ngược lại của size. 
            # Ví dụ, nếu size là (width, height), nó sẽ trả về (height, width).
            return size[::-1] 
        # nếu size không phải 1 list hay 1 tuple thì nó là 1 số duy nhất 
        else: 
            # gọi hàm get_size_with ... để ính toán kích thước mới dựa trên 
            # tỷ lệ khung hình với size được hiểu là chiều dài cạnh nhỏ nhất 
            # max_size là giới hạn kích thước tối đa nếu có 
            return get_size_with_aspect_ratio(image_size, size , max_size)
        
    # gán giá trị size  = get_size 
    size  = get_size(image.size, size, max_size)
    # resize lại hình ảnh với kích thước size các kích thước này được áp dụng 
    # cho hình ảnh được biến đổi trong việc xử lý tăng cường hình ảnh .. 
    rescaled_image = F.resize(image, size)

    # kiểm tra xem giá trị target  = None 
    if target is None: 
        # trả vê recaled_image và 1 giá trị None 
        return rescaled_image , None 
    
    # lấy ra kích thước size của image và rescaled_image và lưu chữ chúng dưới dạng tuple 
    ratios = tuple(float(s) / float(s_orig) for s , s_orig in  zip(rescaled_image.size, image.size))
    # gán cho ratiosh , ratiosw = tuple ratios 
    ratio_width , ratio_height = ratios

    # Thực hiện sao chép danh sách target
    target = target.copy()
    # kiểm tra xem boxes có nằm trong danh sách target 
    if "boxes" in target: 
        # gán boxes = boxes trong target boxes shape [x_max, y_max, x_min, y_min]
        boxes = target["boxes"]
        # tính toán tỷ lệ boxes mới nhân tensor boxes với tensor shape = [ratios_weight, ratios_height, ratios_weight, ratios_height]
        # các giá trị x_max, y_max , x_min , y_min sẽ được nhân với các giá trị tương ứng 
        scaled_boxes = box_xyxy_to_cxcywh * torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        # cập nhật giá trị cho boxes trong danh sách target. 
        target["boxes"] = scaled_boxes

    # kiểm tra xem giá trị của area có tồn tại trong dnah sách target hay không 
    if "area" in target: 
        # gán area bằng giá trị của area trong target
        area = target["area"]
        # tính diện tích của area mới cho các hình ảnh đã được thay đổi size 
        scaled_area = area * (ratio_width * ratio_height)
        # cập nhật lại giá trị của area trong target 
        target["area"] = scaled_area

    # gán h , w = size 
    h, w = size 
    # sau đó cập nhật size = tensor h , w
    target["size"] = torch.tensor([h,w])
    
    # kiểm tra xem mặt nạ có tồn tại trong dnah sách target chưa 
    if "masks" in target: 
        # kêu gọi masks trong target và thực hiện nội suy mask 
        target["masks"] = interpolate(
            # kêu gọi mask và thêm 1 chiều vào masks 
            target['masks'][:, None].float(), size , mode ="neareast"[:, 0] > 0.5
        )
        # trả về rescaled_image, vfa dnah scahs target đã được cập nhật 
        return rescaled_image , target 



# Thiết lập phương thức pad sử dụng để đệm các hình ảnh sau khi được resize 
# nhận đầu vào là hình ảnh , từ điển target và tensor padding chứa giá trị đại diện
# cho số pixel được đệm 
def pad(image, target, padding):
    # giả định rằng chỉ đệm ở góc dưới bên phải 
    # đệm cho hình ảnh với hàm pad truyền vào hình ảnh cần đệm image 
    # với 0, 0 là số pixel đệm cho cạnh trái và phải của hình ảnh ở đây là 0
    # padding[0] giá trị thứ 3 số pixels đệm cho cạnh trên của ảnh  
    # padding[1]: Giá trị thứ tư là số pixel đệm cho cạnh dưới của hình ảnh.
    padded_image = F.pad(image, (0,0, padding[0], padding[1]))
    # kiểm tra xem danh sách target có tồn tại 
    if target is None : # nếu nó rỗng 
        # trả về padded_image và giá trị None 
        return padded_image , None 
    # sap chép lại dnah sách target nếu nó có tồn tại 
    target = target.copy()
    # đảo ngược kích thước của hình ảnh được đệm và gán nó cho giá trị size 
    target["size"] = torch.tensor(padded_image.size[::-1])
    # kiểm tra điều kiện "mask" nếu như mask tồn tại tring danh sacxhs
    if "masks" in target: 
        # nếu nó có tồn tại thực hiện đệm các hình ảnh 
        # ở đây ta chỉ thực hiện đệm cạnh dưới của mặt nạ tương tự như đệm các hình ảnh 
        target["masks"] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    # trả về danh sách các hình ảnh đã được đệm cùng với danh sách target 
    return padded_image , target 



# Thiết lập lớp các phương thức xử lý hình ảnh 
# 1: Random crop cắt ảnh ngẫu nhiên 
class RandomCrop(object):
    def __init__(self, size):   
        # định nghĩa thuộc tính size 
        self.size = size

    def __call__(self, img, target):
        # Thực hiện cắt các hình ảnh với kích thước size 
        region = T.RandomCrop.get_params(img, self.size)
        # sau đó gọi hàm crop để thực hiện áp dụng các 
        # chức năng sử lý hộp giới hạn , mặt nạ của hình ảnh sau khi cắt , và những hình ảnh sau khi được cắt 
        return crop(img, target, region)


# 2: Thiết lập lớp phương thức RandomSizeCrop được sử dụng để cắt các hình ảnh theo mọt kích 
# thước ngẫu nhiên 
class RandomSizeCrop(object):
    # thiết lập phương thức khởi tạo 
    def __init__(self, min_size: int , max_size: int):
        # định nghĩa thuộc tính min_size , max_size 
        self.min_size = min_size
        self.max_size = max_size 
    
    # Thiết lập phương thức call để gọi lại các phuwong thức và tạo quy trình sử lý 
    def __call__(self, img: PIL.Image.Image, target: dict): # đầu vào là hình ảnh và từ điển target
        # tạo ra 2 giá trị ngẫu nhiên là w và h 
        # w là một giá trị ngẫu nhiên từ min_size đến giá trị tối thiểu giữa 
        # w và max_size
        w = random.randint(self.min_size, min(img.width, self.max_size))
        # và h cũng tương tự nhưng với giá trị tối thiểu là h với max_sioze
        h = random.randint(self.min_size, min(img.height, self.max_size))
        # Thực hiện cắt các hình ảnh với kích thuwcs H, W
        region = T.RandomCrop.get_params(img, [h, w])
        # gọi hàm crop để căn chỉnh các hộp giới hạn cho các hình ảnh đã được cắt 
        # mặt nạ cho hình ảnh , và hình ảnh sau khi cắt .. 
        return crop(img, target, region)
    


# 3: CenterCrop lớp phương thức dùng để cắt hình ảnh theo một giá trị trọng tâm 
# được xác định bởi top,left
class CenterCrop(object):
    # thiết lập phương thức khởi tạo 
    def __init__(self, size):
        # định nghia thuộc tính size
        self.size = size 
    
    # tương tự như các lớp phương thức trên xây dựng phương thức call 
    def __call__(self, img, target):
        # lấy ra kích thước hình ảnh đầu vào 
        image_width , image_height = img.size 
        # vào crop_height , crop_width từ thuộc tính size 
        crop_height , crop_width = self.size 
        # tính toán tỷ lệ cắt theo h , w 
        crop_top = int(round((image_height - crop_height) / 2.))
        # tính tỷ lệ cắt left
        crop_left = int(round((image_width - crop_width) / 2.))
        # gọi hàm crop để tính toán hình ảnh sau khi cắt , sau đó trả về hình ảnh 
        # sau khi cắt , danh sách target chứa các giá trị được thay đổi như boxes , masks..
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))
    

# 4 : RadomHorizontalFlip lật ngang ngẫu nhiên 
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        # định nghĩa tỷ lệ lâtj 
        self.p = p
    # THIẾT Lập phương thức call 
    def __call__(self, img, target):
        # kiểm tra xem 1 giá trị random có  < p 
        if random.random() < self.p:
            # nếu có gọi hflip để thực hiện lật ngang hình ảnh 
            # trả về danh sách ảnh đã lật , từ điển target chứa các gía trị boxes , masks 
            # của ảnh đã lật
            return hflip(img, target)
        # trả về từ điển taget và hình ảnh đã được lật ngang của lớp 
        return img, target
    

# 5 : RandomSize lớp phuuwong thức sử dụng để thay đổi kích thước hình 
# ảnh một cahs ngẫu nhiên . 
class RandomSize(object):
    # thiết lập phương thức khởi tạo 
    def __init__(self, sizes, max_size=None):
        # kiểm tra xem size có chắc là  list hay tuple 
        assert isinstance(sizes, (list, tuple))
        # định nghĩa các thuộc tính 
        self.sizes = sizes
        self.max_size = max_size 
    
    # gọi call 
    def __call__(self, img, target=None):
        # gán cho size = 1 gía trị được chọn ngẫu nhiên trong size 
        size = random.choice(self.sizes)
        # trả về hình ảnh đã được resize và từ điển chứa mặt nạ mask , boxes , và diện tích area 
        return resize(img, target, size, self.max_size)
    

# 6 RamdomPad lớp phưuowng thức sử dụng để đệm ngẫu nhiên các hình ảnh 
class RandomPad(object):
    # khởi tạo tham số
    def __init__(self, max_pad):
        # định nghĩa các thuộc tính 
        self.max_pad = max_pad
    # Thiết lập phương thức call 
    def __call__(self, img, target):
        # chọn một giá trị pad_x ngẫu nhiên
        pad_x = random.randint(0, self.max_pad)
        # cà một giá trị pad_y ngẫu nhiên 
        # 2 giá trị này sẽ là số pixel theo nguyên 
        pad_y = random.randint(0, self.max_pad)
        # thực hiện đệm các hình ảnh  truyền vào các tham số cần thiết 
        # kết quả nhận được là các hình ảnh được đệm , và mặt nạ của chúng từ từ điển target
        return pad(img, target, (pad_x, pad_y))


# 7: RandomSelect lớp phương thức sử dụng cho việc lựa chọn phép biến đổi 
# cho hình ảnh 
class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    Chọn ngẫu nhiên các biến đổi 1 và 2 với 1 xác suất cho trước 
    với p cho 1 và 1-p cho 2.
    """
    # định nghĩa phương thưc khởi tại 
    def __init__(self, transforms1, transforms2, p=0.5):
        # định nghĩa phương pháp biến đổi 1 và 2 
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        # và xác suất p 
        self.p = p
    # thiết lập phưuowng thức call 
    def __call__(self, img, target):
        # tạo một xác suất ngẫu nhiên p 
        if random.random() < self.p:
            # nếu thỏa mãn thì áp dụng phép biến đổi 1
            return self.transforms1(img, target)
        # còn không áp dụng phép biến đổi thứ 2
        return self.transforms2(img, target)
    

# 8 : ToTensor chuyển đổi các hình ảnh thành tensor
class ToTensor(object):
    # ở lớp phương thức này không cần xây dựng phương thức định nghĩa init
    def __call__(self, img, target):
        # trả về tensor image và từ điển target 
        return F.to_tensor(img), target


# 9 :  Thiết lập lớp phuuwong thứ RandomErasing để phục vị việc xóa các hình ảnh 
class RandomErasing(object):
    # Định nghĩa phương thức khởi tạo 
    def __init__(self, *args, **kwargs):
        # định nghĩa thuộc tính Eraser với các tham số tự do 
        self.eraser = T.RandomErasing(*args, **kwargs)
    # gọi đến phương thức call để thực hiện tính toán
    def __call__(self, img, target):
        # trả về 
        return self.eraser(img), target


# 10 : Thiết lập lớp xử lý Bình thường hóa hình ảnh 
# thưucj hiện việc phân phối dữ liệu trên pixel để chúng có phân phối đồng nhất hơn
class Normalize(object):
    # thiết lập phuuwong thức khởi tạo nhận đầu vào gồm giá trị mean và std standard diviation
    def __init__(self, mean, std):
        # định nghĩa thuộc tính 
        self.mean = mean
        self.std = std

    # xây dụng phuuwong thức call để tính toán 
    def __call__(self, image, target=None):
        # thực hiện áp dụng việc bình thường hóa cho các hình ảnh
        image = F.normalize(image, mean=self.mean, std=self.std)
        # kiểm tra điều kiện xem dnah sách target = rỗng
        if target is None:
            # nếu rỗng trả về hình ảnh và giá trị None 
            return image, None
        # thực hiện sao chép dnah sách target
        target = target.copy()
        # Lấy ra kích thước h , w từ hình ảnh là 2 kích thước cuối cùng được  
        h, w = image.shape[-2:]
        # kiểm tra xem boxes có trong từ điển target 
        if "boxes" in target:
            # nếu có gán boxes = gái trị boxes trong từ điển target
            boxes = target["boxes"]
            # tính toán trọng tâm của hộp và w ,h của hộp dựa vào boxes [x0, y0, x1, y1]
            boxes = box_xyxy_to_cxcywh(boxes)
            # sau đó chia các gái trị này cho w và 
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            # sau đó cập nhật lại boxes
            target["boxes"] = boxes
        # cuối cùng trả về hình ảnh và từ điển target 
        return image, target
    

# Thiết lập lớp phương thức xử lý trình tự biến đổi 
# được áp dụn cho 2 chuỗi các biến đổi Transformer lên hình ảnh và
# mục tiêu target liên quan 
class Compose(object):
    # Phương thức __init__ khởi tạo lớp với danh sách các biến đổi transforms được truyền vào.
    def __init__(self, transforms):
        self.transforms = transforms

    # Phương thức __call__ cho phép đối tượng của lớp Compose có thể được gọi như một hàm
    #  Khi gọi, nó sẽ lần lượt áp dụng từng biến đổi trong self.transforms lên image và target
    def __call__(self, image, target):
        # . Mỗi biến đổi t được gọi với image và target và trả về cả hai sau khi đã được biến đổi.
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
    # Phương thức __repr__ cung cấp một chuỗi đại diện cho đối tượng Compose, thường được
    # sử dụng để in thông tin chi tiết về đối tượng cho mục đích gỡ lỗi.
    def __repr__(self):
        # Nó liệt kê tất cả các biến đổi trong self.transforms theo một định dạng dễ đọc
        format_string = self.__class__.__name__ + "("
        #Nó liệt kê tất cả các biến đổi trong self.transforms theo một định dạng dễ đọc
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    # Chức năng chính của lớp Compose là để tạo một pipeline biến đổi dễ dàng quản lý và tái sử dụng, giúp việc áp dụng 
    # một loạt các biến đổi trên dữ liệu trở nên thuận tiện và gọn gàng hơn