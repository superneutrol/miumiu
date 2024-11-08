import tensorflow as tf 
import keras 

class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        """
        Tham số Patch_size: à kích thước cho mỗi bản vá hình ảnh được trích 
        suất từ ảnh gốc
        """
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        """
        Thực hiện công việc trích suất các đặc trưng của hình ảnh 
        
        Image_size: shape [batch_size, H, W, C] Example: [64, 64, 64, 3]
        
        mỗi bản vá Patches sẽ được trích suất từ hình ảnh giả patch_size tiêu chuẩn 
        theo bài báo gốc = 16 thì từ ảnh gốc ta có thể tách được 4 * 4 = 16  bản vá

        """
        batch_size = tf.shape(images)[0] # lấy ra kích thước batch_size là chiều đầu tiên 

        # hàm tf.image.extract để trích suất các patches
        patches = tf.image.extract_patches(
            # ảnh nguồn 
            images = images,
            # kích thước của mỗi patch_tensor được chỉ định bởi patch_size
            sizes = [1, self.patch_size, self.patch_size, 1], 
            # kích thước trượt cửa sổ kernel size = một bản vá patch_size * patch_size
            strides = [1, self.patch_size, self.patch_size, 1],
            # rate tỷ lệ đầu ra được lấy từ đầu vào theo các khoảng thời gian được đưa ra bởi rate
            rates = [1, 1, 1, 1], 
            # Valid tức lấy các bản vá đầy đủ trong hình ảnh 
            padding="VALID",
        )
        # lấy ra kích thước của số lượng bản vá = 16 * 16 * 3 = 768
        dim = patches.shape[-1]

        # reshape tensor patches shape = batch_size, -1, dim
        # = shape [batch_size, -1, dim] -1 = (64 * 64) / (16 * 16) = 16
        patches = tf.reshape(patches, (batch_size, -1, dim))
        return patches #
    

class PatchEmbedding(keras.layers.Layer):
    def _init__(self, patch_size, image_size, projection_dim):
        """Tinhs toán số lượng bản vá từ hình ảnh đầu vào có kích thước H * W"""
        super(PatchEmbedding, self).__init__()

        # s = self.num_patches: Number of patches 
        self.num_patches = (image_size // patch_size) ** 2

        # khởi tạo token cls được nhắc đến trong bài báo Vis Transformer token này được 
        # sử dụng để học thông tin hình ảnh sử dụng cho việc phân loại 
        self.cls_token = self.add_weight(
            "cls_token",
            # tensor shape = [1, 1, projection_dim]
            shape =[1, 1, projection_dim], 
            initializer= keras.initializers.RandomNormal(),
            dtype=tf.float32,
        )

        # Đinh nghĩa patches là lớp Patches được sử dụng để tách các bản vá hình ảnh 
        self.patches = Patches(patch_size=patch_size)
        # và một lớp chiếu tuyến tính các hình ảnh 
        self.projection = keras.layers.Dense(units=projection_dim)

        # khởi tạo lớp nhúng vị trí shape [bathc_size, S + 1, D]
        # với S + 1 là số lượng bản vá và bao gồm với biểu diễn token cls 
        self.position_embedidng = self.add_weight(
            "position_embeddings",
            shape=[self.num_patches + 1, projection_dim], 
            initializer = keras.initializers.RandomNormal(),
            dtype=tf.float32,
        )

    def call(self, images):
        """
        Image là 1 danh sách các tensor inputs có shape = [Batch_size, H, W, c]
            VD: example (64, 32, 32, 3)

        mục tiêu trả về 1 tensor encodeed_patches:
            tensỏ này sẽ có shape = [Batch_size, S +1, D]
            shape = (batch_size, S + 1, D) với S = (H, W) // (P ^ 2): Example: (64, 4 + 1, 786)
        """

        # trích xuất các bản vá hình ảnh 
        # patch shape: (..., S, NEW_C) C = 16 * 16 *3 
        patch = self.patches(images)

        # projectioning 
        # => shape [Batch_Size, num_patches, Embedding_dim]
        encoded_patches = self.projection(patch)

        # Extract batch_size
        batch_size = tf.shape(images)[0]
        # and exxtract hidden_size(embedding_size)
        hidden_size = tf.shape(encoded_patches)[-1]
        
        # Sử dụng hàm tf.broadcast để có thể chuyển đổi hình dạng của tensor cls 
        # thành 1 tensor shape (self.cls_token, [batch_size, 1, hidden_size]) mục đihs để có thực hiện phép nối hoặc 
        # phép tính bất kỳ lên nó 
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls_token, [batch_size, 1, hidden_size]),
            dtype= images.dtype,
        )

        # encoded_patches shape: (..., S + 1, D)
        # Thực hiện nối chồng token cls lên đầu tensor biểu diễn điều này sẽ làm cho chiều 
        # thứ 2 tăng kích thước S -> S + 1
        encoded_patches = tf.concat([cls_broadcasted, encoded_patches], axis=1)

        # encoded_patches shape: (..., S + 1, D)
        # Thêm thông tin về nhúng vị trí 
        encoded_patches = encoded_patches + self.position_embedding

        return encoded_patches