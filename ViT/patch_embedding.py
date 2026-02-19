import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embedding_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()

        self.num_patches = (image_size * image_size) // patch_size**2

        self.patch_embedding = nn.Conv2d(in_channels=in_channels,
                                         out_channels=embedding_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         padding=0)

        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)

        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        
        self.embedding_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, C, H, W)
        class_token = self.class_embedding.expand(x.shape[0], -1, -1)

        x = self.patch_embedding(x) # (B, E, H/P, W/P)
        x = self.flatten(x) # (B, E, N)
        x = x.permute(0, 2, 1) # (B, N, E)

        x = torch.cat((class_token, x), dim=1) # (B, N+1, E)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)

        return x