""""Helper script to convert models trained with the main version of DETR to be used with the Detectron2 version.
    Tập lệnh trợ giúp chuyển đổi mô hình đã được huấn luyện với phiên bản chính 
    của DETR để sử dụng với phiên bản Detectron2 . 

"""

import json 
import argparse 
import numpy as np 
import torch
# Note trong tệp mã thì checkpoint là danh sách lưu trữ có thể là 1 từ điển 
# chứa các thông số của mô hình học máy đã được đào tạo , độ lệch (bias) và đôi khi 
# cả thông tin bổ sung như tỷ lệ học , và trạng thái tối ưu hóa 

# Thiết lập phương thức parse_args để xử lý các đối số dùng lệnh 
def parse_args():
    # định nghĩa 1 parse là một trình phân tích đối số có tên là D2 mỏl converter 
    parser = argparse.ArgumentParser("D2 model converter")
    
    # định nghĩa một đường dẫn hoặc URL đến mô hình DETR cần chuyển đổi 
    parser.add_argument("--source_model", default="", type=str, help="Path or url to the DETR model to convert")
    # và một đường dẫn lưu mô hình đã chuyển đổi 
    parser.add_argument("--output_model", default="", type=str, help="Path where to save the converted model")
    # trả về các tham số của parser với parser.parser_args được sử dụng để phân tichd các đối số 
    # được cung cấp từ dòng lệnh. Khi chạy chương trình hàm này sẽ đọc các đối số từ sys.argv 
    # chuyển đổi chúng sang kiểu dữ liệu thích hợp và lưu trữ trong một đối tượng Namspace . 
    return parser.parse_args()



# Hàm main là điểm bắt đầu của script nơi các đối số được phân tích và xử lý 
def main():
    #  xây dựng một tham số args như trên để thực hiện việc phân tích tham số các dòng lệnh 
    args = parse_args()

    # khởi tạo 1 danh sách coco_idx là 1 danh sách các chỉ số lớp từ bộ dữ liệu Coco trong DETR 
    # được huấn luyện . Mục đích của việc này là để ánh xạ các lớp không liên tục sang các lớp 
    # liên tục mà D2 mong đợi 
    coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91]
    # fmt: on   
    # chuyển đổi danh sách này thành ma trận array 
    coco_idx = np.array(coco_idx)
    # kiểm tra xem đường dẫn đến DETR có phải bắt đầu = http 
    if args.source_model.startswith("http"):
        # khởi tạo một biến checkpoint để lưu trữ giá trị checkpoint của DETR được 
        # lấy từ đường dẫn chứa nó 
        checkpoint = torch.hub.load_state_dict_from_url(args.source_model, map_location="cpu", check_hash=True)
    
    # trường hợp còn lại tức đường dẫn source không bắt đầu = http 
    else: 
        # tải tham số DETR xuống nhưng loại trừ check_hash , check hash có chức năng xác minh rằng 
        # tập tin tải xuống chính xác là tập tin mong đợi không bị thay đổi hoặc giả mạo từ nguồn gốc của nó 
        checkpoint = torch.load(args.source_model, map_location="cpu")
    # thêm chuỗi model vào danh sách checkpoint 
    model_to_convert = checkpoint["model"]

    # tạo một từ điển để chứa các tham số chuyển đổi 
    model_converted = {}
    # duyệt qua các khóa cuả từ điển model_to_convert 
    for k in model_to_convert.keys():
        # gán cho các khóa này = old key nghĩa là các khóa của mô hình cũ 
        old_k = k 
        # kiểm tra xem trong khóa k của từ điển có chức tham số là backbone 
        if "backbone" in k : 
            # nếu có thay thế backbone thành 1 chuỗi 
            k = k.replace("backbone.0.body.", "")
            # nếu như layer không tồn tại trong k 
            if "layer" not in k:
                # cộng chuỗi stem vào danh sách k 
                k = "stem." + k
            # duyệt qua 1 danh sách gán các giá trị cho t 
            for t in [1, 2, 3, 4]:
                # thay thế layer tại gia trị t với res lại t + 1 trong danh sahc
                k = k.replace(f"layer{t}", f"res{t + 1}")
            # vơi 1 được gán lần lượt cho 3 giá trị 
            for t in [1, 2, 3]:
                # thực hiện thay thế các kiến trúc trong backbone như trên 
                k = k.replace(f"bn{t}", f"conv{t}.norm")
            # thay thế 1 loạt các giá trị 
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            k = "backbone.0.backbone." + k
        
        # cộng 1 chuỗi detr với danh sách j
        k = "detr." + k
        # in ra màn hình thông tin của k cũ và k mới 
        print(old_k, "->", k)
        # kiểm tra 1 điều kiện nếu như lớp nhúng nằm trong dnah sách k cũ 
        if "class_emb" in old_k: 
            # lấy tensor old_k từ từ điển model_to_convert sau đó sử dụng detach để chuyển chúng 
            # thành tensor mới , detach có chức năng tách gardeint descent 
            # ra khởi quá trình lan chuyển ngược 
            v = model_to_convert[old_k].detach()
            # nếu như dnah sách v có 92 giá trị tức là 92 lớp khác nhau 
            if v.shape[0] == 92: 
                # lấy ra hình dnagj của v 
                shape_old = v.shape 
                # sau đó thêm vào k từ điển model_convert đồng thời gán 
                # theo k một chỉ số indx từ coco_idx 
                model_converted[k] = v[coco_idx]
                # inh ra màn hình console một số thôg tin 
                print("Head conversion: changing shape from {} to {}".format(shape_old, model_converted[k].shape))
                continue
        # gán cho danh sách các k mới trong từ điển model_convert = giá trị old_k từ từ điển 
        # model_to_convert sử dụng detach để tach gradient descent khỏi quá trình lan truyền ngược
        model_converted[k] = model_to_convert[old_k].detach()

    # lưu trữ vào từ điển model với tham số sau xử lý 
    model_to_save = {"model": model_converted}
    # save từ điển này vòa đường dẫn tham số đã xây dựng trước đó 
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()

