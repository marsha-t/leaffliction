from torch import nn
import torch
import math


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, head_nums=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_nums = head_nums
        self.WQ = nn.Linear(self.embed_dim, self.embed_dim)
        self.WK = nn.Linear(self.embed_dim, self.embed_dim)
        self.WV = nn.Linear(self.embed_dim, self.embed_dim)
        self.WO = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.attention_dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim*4),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(self.embed_dim*4, self.embed_dim),
            nn.Dropout(0.1))

    def forward(self, x):
        # layer norm
        B = x.shape[0]
        patch_nums = x.shape[1]
        residual = x
        patches = self.layer_norm1(x)
        # multi head attention
        Q = self.WQ(patches)
        K = self.WK(patches)
        V = self.WV(patches)
        # Multi-head attention splits the Q, K, and V feature dimensions into multiple heads
        # so that the model can learn multiple attention relationships between the patches in parallel.
        Q = Q.reshape(B, patch_nums, self.head_nums, -1)
        Q = Q.permute(0, 2, 1, 3)
        K = K.reshape(B, patch_nums, self.head_nums, -1)
        K = K.permute(0, 2, 1, 3)
        V = V.reshape(B, patch_nums, self.head_nums, -1)
        V = V.permute(0, 2, 1, 3)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        # scores[32, 8, 197,197]
        # the next step for scaling the score so we are avoinding the gradient problems the sqrt of this quantity is equal to the std of the score
        attention = scores/math.sqrt(self.embed_dim//self.head_nums)
        attention = torch.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        output = torch.matmul(attention, V)
        # [32, 8, 197,197] [32,8,197,64]= [32,8,197,64]
        output = output.permute(0, 2, 1, 3)
        # [32, 197, 8,64]

        # collecting the features again together
        output = output.reshape(B, patch_nums, -1)
        # [32, 197, 512]
        # the next step is to let the features that are comming from all the different heads interact
        output = self.WO(output)

        # residual connection
        patches =  residual + output
        residual = patches
        # layer norm
        patches = self.layer_norm2(patches)
        # mlp
        patches = self.mlp(patches)
        # residual connection
        patches = residual + patches
        return patches


class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, channels=3, embed_dim=512, image_size=224, head_nums=8):
        super().__init__()
        self.patch_size = patch_size
        self.head_nums = head_nums
        self.patch_dim = channels * patch_size * patch_size
        self.patch_nums = (image_size//patch_size)**2
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Linear(self.patch_dim, self.embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.POS = nn.Parameter(torch.randn(1, self.patch_nums+1,
                                            self.embed_dim))
        self.classifier = nn.Linear(self.embed_dim, 8)
        self.encoders = nn.ModuleList([
                                      EncoderBlock(embed_dim=self.embed_dim,
                                                   head_nums=self.head_nums)
                                      for _ in range(6)])

    def make_patches(self, images):
        """ take the image and patch it pased on the patch size and return the
          flatten patches """
        B, C, H, W = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3,
                                                                            self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, self.patch_nums, -1)
        return patches

    def forward(self, x):
        B, C, H, W = x.shape
        # 1.
        patches = self.make_patches(x)
        # 2.
        patches_embbedding = self.patch_embeddings(patches)
        #3.
        cls = self.cls.expand(B, -1, -1)
        all_patches = torch.cat((cls, patches_embbedding), dim=1)
        #4
        pos = self.POS.expand(B, -1, -1)
        all_patches = all_patches + pos
        for encoder in self.encoders:
            all_patches = encoder(all_patches)
            cls_token = all_patches[:,0]
            logits = self.classifier(cls_token)
        return logits