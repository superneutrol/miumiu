"""Ploting utilities to visualize training logs."""

import torch 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from pathlib import Path , PurePath 

# Thiết lập phuuwong thức plot_logs sử dụng để vẽ các biểu đồ từ các tệp 
# nhật ký đào tạo 
def log_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP') , 
             ewm_col =0, log_name='log.txt'):
    """Định nghĩa hàm với các tham số:
    logs: danh sách các đường dẫn đến tệp nhật ký
    feilds: các trường dữ liệu cần vẽ biểu đồ 
    ewm_col: cột dùng để làm mịn dữ liệu bằng phương pháp EWM (Exponential Weighted Mean)
    log_name: tên tệp nhật ký, mặc định là 'log.txt'
    """

    # Định nghĩa tên hàm để sử dụng cho việc in thông báo 
    func_name = "plot_utils.py::plot_logs"
    # kiểm tra xem tham số 'logs'có phải là một danh sách các đối tượng path hay không 
    if not isinstance(logs, list):
       # Nếu 'logs' là một đối tượng Path, chuyển nó thành một danh sách
        if isinstance(logs, PurePath):
            logs = [logs]
            # và in ra một thông báo logs đã được chuyển đổi thành 1 danh sách 
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        
        else: #nếu log không phải là danh sách hoặc Path , ném ra một lỗi 
            raise ValueError (f"{func_name} - invalid argument for logs parameter.\n \
        Expect list[Path] or single Path obj, received {type(logs)}")

    # kiểm tra đường dẫn trong danh sách log
    for i , dir in enumerate(logs):
        # Kiểm tra xem mục nàu có phải đối tượng path hay không
        if not isinstance(dir, PurePath):
             raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        # Kiểm tra xem thư mục có tồn tại hay không
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # kiểm tra xem tệp nhật ký có tồn tại trong thư mục hay không 
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return
    
    # tải file nhật ký và vẽ 
    # sử dụng một list comprehension để tạo một danh sách các Dataframe từ các tệp log 
    # mỗi dataFrame được tạo từ một tập nhật ký Json, với mỗi dòng trong tệp Json 
    # trở thành hàng trong DataFrame
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]
    
    # khởi tạo biểu đồ tạo hình vẽ fig và một mảng các trục (axs) sử dụng matplotlib
    # Số lượng cột (ncols) trong mảng trục bằng với số lượng trường dữ liệu cần vẻ. 
    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    # 3: Vẽ biểu đồ :
    # Duyệt qua mỗi dataFrame và mỗi trường dữ liệu. Sử dụng màu sắc từ sns.color_palette
    # để phân biệt các tệp nhật ký khác nhau . 
    for df,color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        # lấy ra giá trị và chỉ số của các trường 
        for j , field in enumerate(fields):
            # kiểm tra xem trường dữ liệu có phải là 'mAP' 
            if field == 'mAP':
                # Tính toán và vẻ biều đồ map 
                coco_eval = pd.DataFrame(
                    # dataa = co_co_dataa_tesst sử dụng stack để nối giá trị theo chiều hàng
                    # và lấy ra tất cả các hàng và cột thứ 2 được chỉ định
                    np.stack(df.test_coco_eval_bbox.dropna().values[:,1])
                ).ewm(com=ewm_col).mean() # sử dụng ewm đê thực hiện cấu hình làm mịn sau đó tính trung bình
                # trọng số mũ dựa trên cấu hình com 
                # tại chỉ số J của trường vẽ biểu đồ theo Fields 
                axs[j].plot(coco_eval, c=color)
            # trường hợp không phải trường 'mAP'
            else:
                # thực hiện nội suy các dataFrame sau đó làm mịn với cấu hình ewm 
                # sau đó tính trung bình trọng số mũ theo cấu hình
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    # sau đó vẽ biểu đồ cho các truwongf khác 
                    y=[f'train_{field}', f'test_{field}'],
                    # class_error: Biểu đồ lỗi phân loại.
                    # loss_bbox_unscaled: Biểu đồ mất mát của bounding box chưa được điều chỉnh tỷ lệ.
                    #mAP: Biểu đồ mean Average Precision,
                    # một chỉ số đánh giá hiệu suất của mô hình phát hiện đối tượng.
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    # duyệt qua danh sách ác biểu đồ và các tên trường
    for ax, field in zip(axs, fields):
        # sử dụng legend để chú thích tên cho các biểu đồ 
        # được trích xuất từ tên của file p trong tập nhật ký
        ax.legend([Path(p).name for p in logs])
        # set tiêu đề cho các trường tương ứng 
        ax.set_title(field)

# thiết lập phương thức plot_precision_recall  
# được định nghĩa để vẻ biểu đồ Precision-Recall và Scores-Recall 
# từ một danh sách các tệp dữ liệu đánh giá mô hình 
def plot_precision_recall(files, naming_scheme='iter'):
    # xác định naming_scheme hàm xác định cách đặt tên cho các biểu đồ . 
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        #  Nếu naming_scheme là 'exp_id' tên sẽ được lấy từ phần thứ 3 từ cuối của đường dẫn tệp
        names = [f.parts[-3] for f in files]
    # nếu iter
    elif naming_scheme == 'iter':
        # tên được lấy từ tên cơ bản (không có phần mở rộng cuả tệp)
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    # khởi tạo biểu đồ subplots 16*5 inch 
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    # vẽ biểu đồ . 
    # duyệt qua các tệp , màu sắc sẽ tương ứng và tên đã xác định 
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        # dữ liệu được tải từ tệp torch.load
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        # precision và scores được lấy dữ liệu và được xử lý để lấy giá trị trung 
        # bình cho tất cả các lớp và khu vực cho 100 phát hiện 
        precision = precision[0, :, :, 0, -1].mean(1)
        # Chọn một lát cắt của mảng precision và scores để lấy dữ liệu
        # chỉ cho IoU đầu tiên (0), tất cả các điểm dữ liệu (:),
        # tất cả các lớp (:), khu vực kích thước đầu tiên (0), và số lượng phát hiện tối đa (-1).
        # Tính giá trị trung bình (mean) qua trục thứ hai (1), tức là trung bình qua tất cả các lớp đối tượng.
        scores = scores[0, :, :, 0, -1].mean(1)
        # prec và rec là giá trị trung bình của Precision và Recall, 
        # được tính toán từ dữ liệu đã xử lý.
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        # Các giá trị này được in ra cùng với thông tin về mAP@50,
        # điểm số trung bình và điểm F1.
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        # Sau đó, biểu đồ Precision-Recall và Scores-Recall được vẽ lên hai trục tương ứng.
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)
    # 
    # : Đặt tiêu đề cho mỗi trục và thêm chú thích để chỉ ra tên tương ứng với mỗi biểu đồ.: Đặt tiêu đề cho mỗi trục
    #  và thêm chú thích để chỉ ra tên tương ứng với mỗi biểu đồ.
    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    # Trả về hình vẽ và trục: Hàm trả về fig và axs, cho phép 
    # người dùng thêm chỉnh sửa hoặc hiển thị biểu đồ sau khi hàm được gọi.
    # Biểu đồ Precision / Recall sẽ hiển thị mối quan hệ giữa độ chính xác và độ nhớ (recall) của mô hình tại các ngưỡng IoU khác nhau.
    # Biểu đồ Scores / Recall sẽ hiển thị mối quan hệ giữa điểm số dự đoán và độ nhớ của mô hình.
    return fig, axs