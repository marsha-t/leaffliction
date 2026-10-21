from torch import nn
import torch
import math


class EncoderBlock(nn.Module):
    """ 
    processes image patch embeddings using multi-head self-attention
    to capture relationships between patches, followed by a feed-forward
    neural network (MLP) to transform the learned representations.
    """
    def __init__(self, embed_dim=512, head_nums=8):
        """
        Initialise the EncoderBlock
        
           Args:
              embed_dim (int): Dimension of each token embedding.
              head_nums (int): Number of attention heads in multi-head self-attention.
        """
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
        """ 
        Applies multi-head self-attention and a feed-forward MLP,
        with layer normalization and residual connections.

        Args:
            x (torch.Tensor): Input token embeddings of shape
                            (batch_size, num_tokens, embed_dim).
        Returns:
            torch.Tensor: Updated token embeddings with the same    
            shape as the input.
       """
        B = x.shape[0]
        patch_nums = x.shape[1]
        residual = x

        patches = self.layer_norm1(x)

        Q = self.WQ(patches)
        K = self.WK(patches)
        V = self.WV(patches)
        Q = Q.reshape(B, patch_nums, self.head_nums, -1)
        Q = Q.permute(0, 2, 1, 3)
        K = K.reshape(B, patch_nums, self.head_nums, -1)
        K = K.permute(0, 2, 1, 3)
        V = V.reshape(B, patch_nums, self.head_nums, -1)
        V = V.permute(0, 2, 1, 3)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        attention = scores/math.sqrt(self.embed_dim//self.head_nums)
        attention = torch.softmax(attention, dim=-1)
        self.last_attention = attention.detach()
        attention = self.attention_dropout(attention)
        output = torch.matmul(attention, V)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(B, patch_nums, -1)
        output = self.WO(output)

        patches =  residual + output
        residual = patches

        patches = self.layer_norm2(patches)

        patches = self.mlp(patches)
   
        patches = residual + patches
        return patches


class VisionTransformer(nn.Module):
    """
        Implements a Vision Transformer (ViT) for image classification.

        Splits images into patches, converts them into embeddings, and adds
        CLS and positional embeddings. The tokens pass through Transformer
        encoder blocks, and the final CLS token is used for classification.
    """
    def __init__(self, patch_size=16, channels=3, embed_dim=512, image_size=224, head_nums=8):
        """
        Initializes the Vision Transformer architecture.

        Args:
            patch_size (int): Size of each square image patch.
            channels (int): Number of input image channels.
            embed_dim (int): Dimension of each token embedding.
            image_size (int): Height and width of the input image.
            head_nums (int): Number of attention heads in each encoder block.
        """
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
        """
        take the image and split it into patches it pased on the patch
        size and return then flatten patches
        Args:
            images(tensor): batch of images 
            """
        B, C, H, W = images.shape
        patches = (images.unfold(2, self.patch_size, self.patch_size)
                        .unfold(3, self.patch_size, self.patch_size))
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, self.patch_nums, -1)
        return patches

    def forward(self, x):
        """
        Performs the forward pass through the Vision Transformer.

        Args:
            images (torch.Tensor): Input images of shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: Classification logits of shape (B, 8).
        """
        B, C, H, W = x.shape
        patches = self.make_patches(x)

        patches_embbedding = self.patch_embeddings(patches)

        cls = self.cls.expand(B, -1, -1)
        all_patches = torch.cat((cls, patches_embbedding), dim=1)

        pos = self.POS.expand(B, -1, -1)
        all_patches = all_patches + pos

        for encoder in self.encoders:
            all_patches = encoder(all_patches)
            cls_token = all_patches[:,0]
            logits = self.classifier(cls_token)
    
        return logits