"""
Modules to compute the maching cost and solve the corresponding LSAP.
    Các modules tính toan chi phí phù hợp và giải LSAP tương ứng . 

"""
import torch 
from scipy.optimize import linear_sum_assignment
import torch 
from torch import nn 
from Utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou 


class HungarianMatcher(nn.Module):
    """
    
    This class computes an asigment between the targets and the predictions of the network
        Lớp này tính toán các phân công giữa các mục tiếu và dự đoán của mạng. 

        For efficiency reasons, the targets don't include the no_object. Because of this , in general,
        there are more predictions than targets. In this case, we do a 1-to-1  maching of the best predictions,
        while the others are un-matched (and thus treated as non-object).


        Vì lý do hiệu quả, các mục tiêu không bao gồm no_object. Bởi vì điều này, trong sự tổng quát 
        có những dự đoán tốt hơn mục tiêu. Trong trường hợp này, Thực hiện đối sánh 1-1 phù hợp cho dự 
        đoán tốt nhất, trong khi những dự đoán khác không khớp (và do đó được coi là không phải đối tượng).

    
    """
    def __init__(self, cost_class: float = 1 , cost_bbox: float = 1, cost_giou: float = 1):
        """
        Creates the matcheer . Tạo trình so khớp 

        Params: 
            Cost_class: This is the relative weight of the classification error in the maching cost .
                        Đây là trọng số tương đối của lỗi phân loại trong chi phí phù hợp. 
            Cost_bbox: This is the relative weight of the L1 error the bounding box coordinates the matching cost. 
                        Đây là trọng số tương đối của lỗi L1 của tọa độ hộp giới hạn trong chi phí phù hợp.
            Cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost. 
                        Đây là trọng số tương đối của lỗi giou của hộp giới hạn trong chi phí phù hợp.
        
        """
        super().__init__()
        # định nghĩa các tham số thuộc tính 
        self.cost_class = cost_class 
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # kiểm tra 1 điều kiện đảm bảo rằng cost_class != 0 hoặc cost_bbox != 0 hoặc 
        # cost_giou != 0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou !=0  , """all cost cant be 0""" # tất cả các cost 
        # không thể bằng không 

    # đặt no_grad để phương thức này bỏ qua việc tính toán gradient descent 
    @torch.no_grad()
    # Phương thức forward sẽ được gọi khi có các tham số chuyền vào 
    def forward(self, outputs, targets):
        """
        Performs the matching. Các biểu diễn phù hợp

        Params: 
            outputs: This is a dict that containts at least these entries: Là một phần lệnh chứa ít nhất các mục này
                Pred_logits: Tnesor of dim [batch_size, num_queries, num_classes] with the classification logits 
                    Tensor shape = [batch_size, num_queries, num_classes] với các nhật ký phân loại.
                Pred_boxes: Tensor of dim [batch_size , num_queries, 4] with predicted box coordinates
                    Tensor shape = [batch_size, num_queries, 4] với tọa độ các hộp được dự đoán. 
                
            targets: This is a list of targets (len(targets) = batch_size), where each target in a dict containing:
                Lables: Tensor of dim [num_target_bboxes] (where num_target_bbox is the number of grounth-truth objects in the taget)
                    containing the class labels 
                Boxes: Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns : 
            A list of size batch_size, containing tuples of (index_i, index_j) where: 
                - index_i is the indices of the selected predictions (in order)
                - index is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: 
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        
        """
        # nén hình dnagj của pred_logits -> shape [batch_size, num_queries * num_classes]
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # we flatten to compute the cost matrices in a batch 
        # chúng ta làm phẳng dồi tính toán ma trận ch phí trong một batch 
        # làm phẳng tensor theo các chiều 0 , 1 thành 1 tensor 2d 
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1) # [batch_size * num_queries, num_classes]
        # tương tự như trên làm phẳng tensor theo các chiều 0 , 1 
        out_bbox = outputs["pred_boxes"].flatten(0, 1) # batch_sze * num_queries, 4

        # nối dnah sách labels trong từ điển target lại với nhau 
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # và nối danh sách các hộp giới hạn trong từ điển target 
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        #Tính toán chi phí phân loại. Ngược lại sự mất mát , không sử dụng NLL , 
        # nhưng xấp xĩ nó trong 1 - proba[target class]
        # 1 là hằng số không thay đổi kết quả phù hợp , nó có thể bỏ qua được 
        cost_class = - out_prob[:, tgt_ids] 

        # Tính toán cho phí L1 giữa các hộp 
        # sử dụng cdist để tính toán khoảng cách giữa các điểm trong hai hộp giới hạn 
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # tính toán iou giữa 2 hộp giới hạn 
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Tính toán ma trận chi phí cuối cùng 
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou 
        # sau đó reshape tensor C shape [bs , num_queries, -1] và phân bổ cho cpu()
        C = C.view(bs, num_queries, -1).cpu()

        # lấy ra số lượng của các hộp giới hạn trong từ điển target 
        sizes = [len(v["boxes"]) for v in targets]
        # Tính toán tối ưu hóa tuyến tính cho các hộp giới hanj
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # trả về danh sách các kết quả sau khi áp dụng phép tối ưu tuyến tính 
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)