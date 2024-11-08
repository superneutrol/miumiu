from colossalai.zero.shard_utils import TensorShardStrategy # lớp này liên quan đến một kỹ thuật tối ưu hóa
# hoặc chiến lược quản lý bộ nhớ 

# cấu hình một từ điển zero 
zero = dict(
    # bên trong từ điển này có một khóa là model_config ánh xạ đệ 
    # đến một từ điển khác 
    model_config = dict(
        # từ điển này gồm 3 khóa 
        # shard_strategy: Khóa này được liên kết với một thể hiện của lớp TensorShardStrategy
        shard_strategy = TensorShardStrategy(),
        tensor_placement_policy = 'cpu',
        # reuse_fp16_shard: Khóa này được đặt là False. 
        # Nó có thể điều khiển việc mô hình có tái sử dụng các phần (shards) của tensor dạng số thực 16-bit (FP16) hay không.
        reuse_fp16_shard = False
    )
)
# gradient_accumulation: Biến này được đặt là 4. Nó có thể đại diện cho số bước 
# tích lũy gradient trước khi thực hiện cập nhật trọng số trong quá trình huấn luyện.
gradient_accumulation = 4
# clip_grad_norm: Biến này được đặt là 1.0. Nó có thể được sử dụng để cắt tỉa gradient, 
# đảm bảo rằng gradient không vượt quá một chuẩn nhất định trong quá trình lan truyền ngược.
clip_grad_norm = 1.0