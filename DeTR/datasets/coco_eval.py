# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import Utils 
from Utils.misc import all_gather

# Xây dựng lớp xử lý CocoEvaluator để xây dựng bộ dữ liệu Coco cho việc đánh giá mô hình 
# lớp CocoEvaluator được sử dụng để đánh giá mô hình phát hiện đối tượng dựa trên dữ liệu Coco
# và các loại chỉ số IOU khác nhau . 
class CocoEvaluator(object):
    # Xây dựng phương thức khởi tạo  coco_gt là ground truth của hình ảnh và iou_types
    def __init__(self, coco_gt, iou_types):
        # và đảm bảo rằng iou_types là một list hoặc 1 tuple nếu không ném ra một lỗi 
        assert(iou_types, (list, tuple))
        # sử dụng deepcopy để sao chéo sâu đối tượng coo_gt để khi thực hiện các thay đổi trên 
        # bản sao , đối tượng gốc không ảnh hưởng . 
        coco_gt = copy.deepcopy(coco_gt)
        # đingh nghĩa thuộc tính coco_gt 
        self.coco_gt = coco_gt 

        # và thuộc tính iou_types 
        self.iou_types = iou_types
        # định nghĩa 1 từ điển coco_eval , mỗi đối tượng trong từ điển sẽ đánh giá 1
        # loại IOU khác nhau. 
        self.coco_eval = {}
        # duyệt qua danh sách iou_types chứa các loại IOU khác nhau 
        for iou_type in iou_types:
            # áp dụng hàm CocoEval cho coco_gt và IOU_type sau đó thêm nó vào từ điển 
            self.coco_eval[iou_type] = COCOeval(coco_gt, iou_type)
        
        # một dnah sách chưá các IDS của hình ảnh 
        self.img_ids = []
        # và một từ điển eval_imgs lưu trữ danh sách các giá trị iou khác nhau 
        self.eval_imgs ={k: [] for k in iou_type}
    
    # Thiết lập phương thức update để có thể thực hiện các  cập nhật 
    def update(self, predictions):
        # predictions là một từ điển 
        # lấy các giá trị khóa duy nhất của dự đoán trong predictions dưới dạng danh sách 
        # và gan snos cho img_ids 
        img_ids = list(np.unique(list(predictions.keys())))
        # sử dụng extend để nối cac các danh scahs này với nhau và thêm nó vào 
        # dnah sách lưu trữ img_ids 
        self.img_ids.extend(img_ids)

        # duyệt qua dnah scahs chứa các iou khác nhau 
        for iou_type in self.iou_types: 
            # sử dụng hàm prepare với tham số iou để quyết định xem kết 
            # quả trả về của iou_types này thuộc loại phân đoạn hình ảnh (segment), bounding_boxes, và keypoints 
            results = self.prepare(predictions, iou_type)


            # ngăn chặn các bản in pycocotools 
            # dùng context manager để tạm thời chuyển hướng tới stdout tới /dev/null 
            # nhằm mục đích ngăn chặn việc in ra từ thư viện pycocotools 
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    # tải kết quả dự đoán vào coco đối tượng nếu có kết quả
                    # nếu không có kết quả tạo ra một đối tượng COco mới 
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

            # GÁN LẠI COCO_EVAL =  LOẠI IOU HIỆN TẠI 
            coco_eval = self.coco_eval[iou_type]

            # gán đối tươngj coco với kết quả dự đoán vào COCOeval 
            coco_eval.cocoDt = coco_dt
            # cập nhật tham số id hình ảnh trong tham số của Cocoeval 
            coco_eval.params.imgIds = list(img_ids)
            # thực hiện đnahs giá và trả về danh sách ID hình ảnh và kết quả đánh giá
            img_ids, eval_imgs = evaluate(coco_eval)

            # thêm kết qủa đánh giá vào dnah sách tương ứng với loại IOU 
            self.eval_imgs[iou_type].append(eval_imgs)


    # Xây dựng phương thức để đồng bộ hóa các tiến trình 
    # đảm bảo rằng tất cả các quy trình đều được xem xét giúp cho việc đánh giá kết quả cuối cùng trở nên toàn diện và chính xác hơn
    # ở đây thực hiện đồng bộ hóa các kết quả đánh giá giữa các quá trình 
    def synchronize_between_processes(self):
        # duyệt qua danh sách chứa các loại IOU khác nhau 
        for iou_type in self.iou_types:
            # gán cho iou_type trong dnah sách eval_image bằng 
            # kết quả của phép nối các mảng đánh giá lại với nhau theo chiều thứ 3
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            # tạo đánh gía coco chung cho tất cả các quá trình 
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    
    # Thiết lập phuuwong thức acimulate tích lũy kết quả đánh giá từ các đối tượng COCOeval khác nhau. 
    # Tích lũy là một bước quan trọng trong qúa trình đánh giá, nơi các kết quả từ các hình ảnh 
    # khác nhau được tổng hợp lại để tạo một đánh giá tổng thể 
    def accumulate(self):
        # Duyệt qua từng đối tượng COCOeval trong từ điển
        for coco_eval in self.coco_eval.values():
            # Tích lũy kết quả đánh giá
            coco_eval.accumulate()
    
    # in ra và toms tắt kết quả đánh giá cho mỗi loại IOU. bao gồm việc hiển thị các chỉ số đánh 
    # giá chính 
    def summarize(self):
        # Duyệt qua từng cặp loại IOU và đối tượng COCOeval
        for iou_type, coco_eval in self.coco_eval.items():
            # In ra loại IOU đang được xem xét
            print("IoU metric: {}".format(iou_type))
            # Tóm tắt kết quả đánh giá
            coco_eval.summarize()

    # phưuowng thức prepare để thực hiện chuẩn bị dữ liệu cho loại kiểu dữ 
    # liệu coco
    def prepare(self, predictions, iou_type):
        # nếu iou_type là bounding_box
        if iou_type == "bbox":
            # nếu là hộp giới hạn trả về chuẩn bị kết quả cho coco detection
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            # tương tự như trên nhưng trả về sự chuẩn bị cho phân đoạn hình ảnh 
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            # và các khóa điểm 
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    # chuẩn bị cho coco trong nhiệm cụ detection
    def prepare_for_coco_detection(self, predictions):
        # tạo một danh sách chưa kết quả 
        coco_results = []
        # duyệt qua dnah scahs các dự đoán trả về id và giá trị dự đoán hay 
        # là id của hình ảnh được dữ đoán và kết quả của nó 
        for original_id, prediction in predictions.items():
            # kiểm tra xem danh sách prediction có tồn tại
            if len(prediction) == 0:
                # nếu bằng 0  tiếp tục 
                continue
            # lấy ra hộp giới hạn từ dự đoán
            boxes = prediction["boxes"]
            # chuyển đổi hộp giới hạn thành 1 dạn sách chứa 
            # tọa độ x,y , h, w 
            boxes = convert_to_xywh(boxes).tolist()
            # lấy ra điểm score của dự đoán và chuyển nó thành dnah sách
            scores = prediction["scores"].tolist()
            # tương tư với nhãn labels 
            labels = prediction["labels"].tolist()

            # nối các kết quả bằng hàm mở rộng extend 
            coco_results.extend(
                [
                    { #gồm id hình ảnh 
                        "image_id": original_id,
                        # nhãn của đối tượng theo chỉ số của hộp giới hạn tương ứng
                        "category_id": labels[k],
                        # hộp giới hạn , điểm score
                        "bbox": box,
                        "score": scores[k],# tương tự như trên
                    }
                    # duyệt qua các hộp giới hạn có trong ảnh lấy ra giá trị và chỉ số
                    for k, box in enumerate(boxes)
                ]
            )
        # trả về danh scahs cuôi cùng chứa các tham số cho nhiệm vụ phát hiện đối tượng
        return coco_results
    
    # Thực hiện chuẩn bị dữ liệu cho việc phân đoạn thực thể
    def prepare_for_coco_segmentation(self, predictions):
        # tạo một dnah sách chứa kết qủa đầu ra cuối cùng 
        coco_results = []
        # lấy gia id của hình ảnh và giá trị của hình ảnh được dự đoán từ dnah sác dự đoán
        for original_id, prediction in predictions.items():
            # kiểm tra nếu như giá trị predict = 0
            if len(prediction) == 0:
                # tiếp tục lặp 
                continue

            # lấy ra cấc giá trị score từ đầu ra dự đoán 
            scores = prediction["scores"]
            # nhãn của đối tượng
            labels = prediction["labels"]
            # và mặt nạ biểu diễn 
            masks = prediction["masks"]

            # gán cho mask > 0.5 tức là chuyển mặt nạ snag nhị phân dự trên nguongwx 0.5
            masks = masks > 0.5

            # chuyển đổi accs giá trị score và label thành list 
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            ## Mã hóa mặt nạ sang định dạng RLE để sử dụng trong COCO
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            # giải mã các chuỗi counts từ bytes sang utf8
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            # mở rộng danh sách kết quả bằng cách thêm thông tin dự đoán z
            coco_results.extend(
                [
                    {
                        # chỉ số ids 
                        "image_id": original_id,
                        # id phân loại cho đối tượng trong hình ảnh 
                        "category_id": labels[k],
                        # segmentations với values là các chuỗi cuounts đã được giair mã utf8
                        "segmentation": rle,
                        # và điểm score cho chỉ số k của segmentations 
                        # segmectations là một danh sách chứa các polysgon mỗi polysgon chứa các điểm biểu diễn đa giác 
                        # cho thực thể có trong hình ảnh , mỗi đa giác là một hộp giới hạn
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        # Trả về danh sachcs kết qủa
        return coco_results
    
    # Thiết lập phương thức chuẩn bị dữ liệu dự đoán cho định dạng COCO Keypoint
    # là 1 chuẩn  phổ biến cho việc đánh dấu và đánh giá các mô hình thị giá máy tính
    def prepare_for_coco_keypoint(self, predictions):
        # tạo một danh sách mới để lưu trữ các kết quả dưới dạng COCO keypoint 
        coco_results = []
        # duyệt qua từng dự đoán trong từ điển 
        for original_id, prediction in predictions.items():
            # kiểm tra xem nếu như dự đoán = rỗng 
            if len(prediction) == 0:
                continue
            
            #Lấy ra các thông tin cần thiết từ dự đoán.
            # hộp giới hạn cho mỗi đối tượng được phát hiện 
            boxes = prediction["boxes"]
            # chuyển đổi hộp giới hạn sang định dạng x, y  w, h
            boxes = convert_to_xywh(boxes).tolist()
            # điểm số đánh giá mức độ chắc chắn của dự đoán
            scores = prediction["scores"].tolist()
            # nhãn của đối tượng được phát hiện 
            labels = prediction["labels"].tolist()
            # các điểm then chốt được dự đoán 
            keypoints = prediction["keypoints"]
            # làm phẳng vaà chuyển đổi nó thành danh scahs 
            keypoints = keypoints.flatten(start_dim=1).tolist()

             # Tạo danh sách kết quả COCO Keypoint từ các thông tin đã lấy ra.
            coco_results.extend(
            [
                {
                    "image_id": original_id,  # ID của ảnh gốc.
                    "category_id": labels[k],  # Nhãn của đối tượng.
                    'keypoints': keypoint,  # Các điểm then chốt.
                    "score": scores[k],  # Điểm số của dự đoán.
                }
                for k, keypoint in enumerate(keypoints)  # Duyệt qua từng bộ điểm then chốt.
            ]
        )
        return coco_results  # Trả về danh sách kết quả.



# thiết lập phương thức convert_to_xywwh để chuyển đổi 
# các hộp giới hạn vê định dnagj x, y , w , h
def convert_to_xywh(boxes):
    # sử dụng hàm unbind để phân tách các chiều của tensor boxes 
    # và loại bỏ  chiều thứ 2 tức chiều chứa các giá trị biểu diễn bonding boxes
    xmin, ymin, xmax, ymax = boxes.undind(1)
    # trả về định dạng xmin, ymin w , h
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

# xây dựng phuuwong thức merget đê thực hiện hợp nhất các danh sách hình ảnh
# được dự đoán
def merge(img_ids, eval_imgs):
    # gọi hàm all_gather để gôm danh sách img_ids và eval_image
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    # tạo 1 danh sách merged_img_ids để sáp nhật danh sách img lại với nhau
    merged_img_ids = []
    # duyệt qua các phần tử năm trong danh sách all_img_ids
    for p in all_img_ids:
        # sử dụng extend để nối chúng và thêm vào danh sách đã tạo 
        merged_img_ids.extend(p)

    # tương tự như trên 1 danh sách chứa val_image
    merged_eval_imgs = []
    # duyệt qua các phần tử nằm trong danh sách
    for p in all_eval_imgs:
        # thêm nó vào danh sachs chứa đã tạo 
        merged_eval_imgs.append(p)

    # chuyển đổi danh sách các img_ids thành array 
    merged_img_ids = np.array(merged_img_ids)
    # và danh sách eval thành mảnh danh sahc được nối theo chiều thứ 3
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    # đặt các hình ảnh duy nhất trong image_ids và xắp xếp giảm dần 
    # việc này nhằm tránh việc trùng lặp các hình ảnh 
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    # sau đó ta lấy 1 lát cắt theo chỉ số của ixd theo chiều thứ 3 
    merged_eval_imgs = merged_eval_imgs[..., idx]

    # cuối cùng trả về 2 danh sách đã được hợp nhất 
    return merged_img_ids, merged_eval_imgs



# Thiết lập phương thức chung khởi tạo bộ xác thực coco
def create_common_coco_êval(coco_eval, img_ids , eval_imgs):
    # thực hiện việc hợp nhất các dnah sách
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    # chuyển đổi img_ids thành 1 danh sahcs
    img_ids = list(img_ids)
    # tương tự eval thành 1 danh sách được làm phẳng thành mảng 1 chiều 
    eval_imgs = list(eval_imgs.flatten())

    # Gán list eval_imgs đã làm phẳng vào thuộc tính evalImgs của đối tượng coco_eval.
    coco_eval.evalImgs = eval_imgs
    # Gán list img_ids vào thuộc tính imgIds của đối tượng params trong coco_eval.
    coco_eval.params.imgIds = img_ids
    # Tạo một bản sao sâu của coco_eval.params và gán nó vào thuộc tính _paramsEval.
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(self):

    """"Hàm này thực hiện các bước sau:
    1: Thiết lập loại đánh giá dựa trên tham số useSegm.
    2: Chuẩn bị danh sách ID ảnh và danh mục không trùng lặp.
    3: Chuẩn bị dữ liệu cần thiết cho việc đánh giá.
    4: Tính toán IoU (Intersection over Union) hoặc OKS (Object Keypoint Similarity) cho từng ảnh dựa trên loại đánh giá.
    5: Đánh giá từng ảnh và lưu trữ kết quả vào một mảng numpy đã được làm phẳng.
    6: Sao chép tham số đánh giá để sử dụng sau này.
    7: Hàm này không in ra thông tin trong quá trình thực hiện nhưng trả về 
    danh sách ID ảnh và kết quả đánh giá để có thể sử dụng cho các bước tiếp theo trong quá trình đánh giá."""
    # Định nghĩa hàm evaluate, không trả về giá trị.
    p = self.params  # Lấy tham số từ đối tượng hiện tại.

    # Kiểm tra và thiết lập loại đánh giá dựa trên tham số useSegm.
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
    
    # Chuẩn bị danh sách ID ảnh và danh mục, loại bỏ trùng lặp và sắp xếp.
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p  # Cập nhật tham số.

    self._prepare()  # Chuẩn bị dữ liệu cần thiết cho việc đánh giá.

    # Xác định danh sách ID danh mục.
    catIds = p.catIds if p.useCats else [-1]

    # Chọn hàm tính IoU phù hợp với loại đánh giá.
    if p.iouType in ['segm', 'bbox']:
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks

    # Tính IoU cho từng cặp ID ảnh và ID danh mục.
    self.ious = {(imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds}

    # Đánh giá từng ảnh và lưu trữ kết quả.
    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet) for catId in catIds for areaRng in p.areaRng for imgId in p.imgIds]

    # Chuyển đổi evalImgs thành mảng numpy và làm phẳng theo chiều cần thiết.
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))

    # Sao chép tham số đánh giá để sử dụng sau này.
    self._paramsEval = copy.deepcopy(self.params)

    # Trả về danh sách ID ảnh và kết quả đánh giá.
    return p.imgIds, evalImgs