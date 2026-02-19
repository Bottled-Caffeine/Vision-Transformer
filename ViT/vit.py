from torch import nn
from patch_embedding import PatchEmbedding
from transformer_encoder_block import TransformerEncoderBlock

class ViT(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embedding_dim: int = 768,
                 embedding_dropout: float = 0.1,
                 num_heads: int = 12,
                 attn_dropout: float = 0,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 num_layers: int = 12,
                 num_classes=1000):
        super().__init__()

        assert image_size % patch_size == 0, \
            "Image size must be divisible by patch size"

        assert embedding_dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads"
        
        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              in_channels=in_channels,
                                              embedding_dim=embedding_dim,
                                              dropout=embedding_dropout)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           attn_dropout=attn_dropout,
                                                                           mlp_dropout=mlp_dropout) for _ in range(num_layers)])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)

        return self.classifier(x[:, 0])