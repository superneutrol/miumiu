from tensorflow.python.keras.layers.core import Dropout
import Vit 
from Vit.embedding import PatchEmbedding 
from Vit.encoder import TransformerEncoder 
import keras, tensorflow
from keras import layers 


class VIT(keras.models.Model):
    def __init__(self, num_layers=2, num_heads=12,
                D=768, mlp_dim=3070, num_classes=10,
                patch_size=16, image_size=224, dropout=0.1,
                norm_eps=1e-6):
        super(VIT, self).__init__()
        """
        VIsionTransformer Model
            Parameters
            ----------
            num_layers: int,
                number of transformer layers
                Example: 12
            num_heads: int,
                number of heads of multi-head attention layer
            D: int
                size of each attention head for value
            mlp_dim: 
                mlp size or dimension of hidden layer of mlp block
            num_classes:
                number of classes
            patch_size: int
                size of a patch (P)
            image_size: int
                size of a image (H or W)
            dropout: float,
                dropout rate of mlp block
            norm_eps: float,
                eps of layer norm
        """
        # áp dụng việc tăng cường dữ liệu hình 
        self.data_augmentation = keras.Sequential(
            [
            keras.layers.Rescaling(scale= 1. / 255), # [0 -> 1]
            keras.layers.Resizing(image_size, image_size) ,# image_shaoe {224 * 224}
            # Lật ngược ảnh 
            keras.layers.RandomFlip("honrizontal"), 
            keras.layers.RandomRotation(factor=0.02),
            keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ])

        # Nhúng các bản vá hình ảnh 
        self.embeding = PatchEmbedding(patch_size, image_size, D)

        # Encoder Transformer 
        self.encoder = TransformerEncoder(
            num_heads=num_heads, 
            num_layers=num_layers, 
            D=D, 
            mlp_dim=mlp_dim, 
            dropout=dropout, 
            norm_eps=norm_eps, 
        )

        # MultiLinear header 
        self.mlp_head = keras.Sequential([
            keras.layers.Normalization(epsilon=norm_eps),
            keras.layers.Dense(mlp_dim), 
            keras.layers.Dropout(dropout),
            keras.layers.Dense(num_classes, activation="softmax"),
        ])

        self.last_layer_norm = keras.layers.LayerNormalization(epsilon=norm_eps)


    def call(self, inputs):
        # create augmented data 
        # augmented shape (.., image_size, image_size, channels)
        augmented = self.data_augmentation(inputs)

        # Create position embedding + CLS token 
        # shape [ batch_size, num_elements + 1, embedim]
        embedded = self.embeding(augmented)

        # encoder patch with Transformer Encoder 
        encoded = self.encoder(embedded)

        # Embeded_cls , shape [.., D]
        # lấy tất cả chiều đầu và các phần tử đầu tiên của batch bỏ qua chiều cuối cùng
        embedded_cls = encoded[:, 0]
        # Last layer norm
        y = self.last_layer_norm(embedded_cls)
        
        # Feed MLP head
        # output shape: (..., num_classes)

        output = self.mlp_head(y)

        return output


class VitBase(VIT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, 
                 dropout=0.1, norm_eps=1e-6):
        super().__init__(num_layers=12, num_heads=12, D=768, mlp_dim=3072,
                         num_classes=num_classes, patch_size=patch_size, 
                         image_size=image_size, dropout=dropout, norm_eps=norm_eps)
        
    

class ViTLarge(VIT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=24,
                         num_heads=16,
                         D=1024,
                         mlp_dim=4096,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)


class ViTHuge(VIT):
    def __init__(self, num_classes=10, patch_size=16, image_size=224, dropout=0.1, norm_eps=1e-12):
        super().__init__(num_layers=32,
                         num_heads=16,
                         D=1280,
                         mlp_dim=5120,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         image_size=image_size,
                         dropout=dropout,
                         norm_eps=norm_eps)