import json 
import os 
import Utils 
# kêu gọi hàm misc là một tập hợp các phương thức bao gồm 
# xây dựng môi trường phân tán , các quy trình tham gia vào sử lý ...
import Utils.misc as utils 

# xây dựng 1 khối ngoại lệ try: 
try : 
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass

class PanopticEvaluator(object):
    # thiết lập phương thức khởi tạo và định nghĩa các tham số 
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        # khởi tạo các biến thành viên của class 
        # đường dẫn tới file annotation (chú trhichs)ground truth. 
        self.gt_json = ann_file 
        # Thư mục chứa các file ảnh ground truth 
        self.gt_folder = ann_folder 
        # kiểm tra xem đây có phải là quy trình chính không 
        if utils.is_main_process():
            # tạo thư mục output nếu nó không tồn tại 
            if not os.path.exists(output_dir):
                # tạo thư mục output 
                os.mkdir(output_dir)
        
        # định nghĩa output_dir 
        self.output_dir = output_dir
        # là danh sách để lưu trữ các dự đoán predictions 
        self.predictions = []
    
    # thiêt lập phương thức update được sử dụng để cập nhật danh sách dự đoán
    # và lưu các file ảnh dự đoán
    def update(self, predictions):
        # duyệt qua dah sách dự đoán 
        for p in predictions: 
            # mở file theo đường dẫn và tên file được chỉ định trong predictions
            # và gi dữ liệu ảnh 
            with open(os.path.join(self.output_dir, p['file_name'], 'wb')) as f : 
                # ghi dữ liệu vào cuối file ảnh 
                f.write(p.pop("png_string"))
        
        # thêm các dự đoán mới vào danh sách prediction 
        self.gt_json += predictions 

    # xây dựng phương thức đồng bộ hóa các dự đoán từ nhiều quy trình 
    # quy trình này đảm bảo rằng tất cả các dự đoán từ các quy trình khác nhau 
    # được xem xet , giúp cho việc đánh giá kết quả cuối cùng trở nên toàn diện và chính xác hơn
    def synchronize_between_processes(self):
        # Gọi hàm 'all_gather' từ module 'utils' để thu thập các dự đoán từ tất cả các quy trình.
        # hàm all_gather được sử dụng để tập trung lại tất cả các giá trị dự đoan trên 
        # tất cả các tiến trình của quá trình trong môi trường phan tán 
        all_predictions = utils.all_gather(self.predictions)
        # Khởi tạo một danh sách mới để chứa các dự đoán đã được hợp nhất.
        merged_predictions = []
        # Duyệt qua danh sách các dự đoán thu thập được và hợp nhất chúng.
        # và gán nó cho p 
        for p in all_predictions:
            # thêm p vào danh sách marged 
            merged_predictions += p
        # Cập nhật danh sách dự đoán của đối tượng với danh sách dự đoán đã được hợp nhất.
        self.predictions = merged_predictions


# Phương thức summarize trong class PanopticEvaluator có chức năng tạo ra một tóm tắt
# của các dự đoán và tính toán các chỉ số đánh giá panoptic
def summarize(self):
    # Kiểm tra xem đây có phải là quy trình chính không.
    if utils.is_main_process():
        # Tạo một dictionary chứa các dự đoán.
        json_data = {"annotations": self.predictions}
        # Tạo đường dẫn tới file JSON sẽ chứa các dự đoán.
        predictions_json = os.path.join(self.output_dir, "predictions.json")
        # Mở file và ghi dữ liệu JSON vào đó.
        with open(predictions_json, "w") as f:
            f.write(json.dumps(json_data))
        # Gọi hàm 'pq_compute' để tính toán các chỉ số đánh giá panoptic.
        return pq_compute(self.gt_json, predictions_json, gt_folder=self.gt_folder, pred_folder=self.output_dir)
    # Nếu không phải quy trình chính, không làm gì cả và trả về None.
    return None

    # phương thức sẽ tạo một file JSON mới chứa các dự đoán đã được lưu trong self.predictions.
    # Sau đó, nó sử dụng hàm pq_compute (không được định nghĩa trong đoạn mã này)
    # để tính toán các chỉ số đánh giá panoptic dựa trên file JSON của dự đoán và file JSON của ground truth.
    # Cuối cùng, nếu không phải quy trình chính, phương thức sẽ trả về None,
    # có nghĩa là không có hành động nào được thực hiện.