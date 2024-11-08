from torch.optim import AdamW, Adam

# Thiết lập phương thức separate_weight_decayble_params được sử dụng 
# để có thể phân dã các tham số và chia chúng 
def separate_weight_decayable_params(params):
    # khởi tạo 2 dnah sách dỗng để lưu trữ weight_decay và no_weight_decay 
    wd_params, no_wd_params = [], []
    # Duyệt qua danh sách tham số params và lấy ra từng tensor một 
    for param in params: 
        # nếu như số chiều của tensor param < 2 thì ta lấy 
        # no_weight_decay avf ngược lại dồi gán cho param_list là một danh sách chứa các tham số 
        param_list = no_wd_params if param.ndim < 2 else wd_params 
        # thêm các gia trị của tensor param tương ứng aào với danh sách param_list 
        param_list.append(param)

    # trả về 2 danh sách weight_decay và no_weight_decay 
    return wd_params, no_wd_params


# Xây dựng trình tối ưu hóa cho mô hunhf 
def get_optimizer(
        params, lr=1e-4, wd = 1e-2, betas = (0.9, 0.99),
        eps = 1e-8, filter_by_requires_grad= False, 
        grounp_wd_params = True, **kwargs 
): 
    # nếu như gradient yêu cầu một bộ lọc filter 
    if filter_by_requires_grad: 
        # áp dụng một bộ lên tham số với số lần lọc bằng với số 
        # lượng phần tử trong tensor params 
        params = list(filter(lambda t : t.requires_grad, params))

    # nếu như tỷ lệ phân dã weight_decay = 0
    if wd == 0: 
        # ta xây dựng trình tối ưu hóa adam 
        return Adam(params, lr=lr, betas = betas, eps= eps)
    
    # Nếu như tham số chỉ định nhóm các trọng số phân dã = True 
    if grounp_wd_params: 
        # áp dụng việc phân tách các nhóm trọng số có sử dụng weight_decay và no_weight_decay 
        # kết quả là 1 tuple 
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        # và sau đó khởi tạo nên 2 từ điển chứa chúng và lưu các từ điển 
        # này vào dnah sách 
        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    # trường hợp cuối cùng khởi tạo và trả về trình tối ưu hóa ADAMW
    return AdamW(params, lr = lr, weight_decay = wd, betas = betas, eps = eps)