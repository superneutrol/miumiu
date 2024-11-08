# Thiết lập các cấu hình epoch, weight_decay, learning_rate 
# là các tham số dạng hằng số sử dụngcho quá trình đào tạo và tối ưu hóa mô hình 
EPOCHS = 1 
LEARNING_RATE = 0.001 
WEIGHT_DECAY = 1e-2 

# và một hệ số tích lũy GRADIENT 
gradient_accumulation = 1 
# và một tỷ lệ cắt xén tham số cho gradent 
clip_grad_norm = 0.0 