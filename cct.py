# imports
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


# default config
BATCH_SIZE = 32
IN_CHANNELS = 3
IMG_SIZE = 224
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
POOLING_KERNEL_SIZE = 3
POOLING_STRIDE = 2
POOLING_PADDING = 1
# CONV_BLOCK_DROPOUT = 0.
PATCH_SIZE = 16
EMBED_TYPE = 'sinusoidal' # None, learnable, sinusoidal
EMBED_DIM = 768
EMBED_DROPOUT = 0
NUM_MSA_HEADS = 12
ATTN_DROPOUT = 0
MLP_HIDDEN_DIM = EMBED_DIM * 4
MLP_DROPOUT = 0.1
NUM_CONV_LAYERS = 1
NUM_VIT_LAYERS = 7
NUM_CLASSES = 10
CLASSIFIER_DROPOUT = 0.1


# util functions
def cnn_input_tensor_validation(input_tensor:torch.Tensor,
                                in_channels:int) -> None:
    """Function that asserts correct dimensionality CNN input tensor, and that the channel count is correct.
    
    ARGS
        input_tensor (torch.Tensor): the tensor passed into the forward pass method
        patch_size (int): the size of the ViT patches"""

    # check dimensionality of input tensor
    input_shape = input_tensor.shape
    assert len(input_shape) == 4, f"[Error] input shape must be 4D [B C H W], got {len(input_shape)}D input."

    # check channel count is correct
    tensor_channels = input_tensor.shape[1]
    assert tensor_channels == in_channels, f"[Error] input tensor contained {tensor_channels} channels, expected {in_channels}."


def vit_input_tensor_validation(input_tensor:torch.Tensor,
                                patch_size:int) -> None:
    """Function that asserts correct dimensionality ViT input tensor, and that image shape is divisible by patch_size
    
    ARGS
        residual_stream (torch.Tensor): the tensor passed into the forward pass method
        patch_size (int): the size of the ViT patches"""

    # check dimensionality of input tensor
    input_shape = input_tensor.shape
    assert len(input_shape) == 4, f"[Error] input shape must be 4D [B C H W], got {len(input_shape)}D input."

    # check that height and width are divisible by patch size
    img_height = input_shape[-2]
    assert_msg = f"[Error] Image height ({img_height}) must be divisible by patch size ({patch_size})."
    assert img_height % patch_size == 0, assert_msg

    img_width = input_shape[-1]
    assert_msg = f"[Error] Image width ({img_width}) must be divisible by patch size ({patch_size})."
    assert img_width % patch_size == 0, assert_msg


def residual_stream_validation(residual_stream:torch.Tensor,
                               embed_dim:int,) -> None:
    """Function that asserts correct dimensionality and embedding shape of transformer residual stream
    
    ARGS
        residual_stream (torch.Tensor): the tensor passed into the forward pass method
        embed_dim (int): the hidden dimension of the transformer"""
   
    # validate dimensionality of residual stream
    input_shape = residual_stream.shape
    assert len(input_shape) == 3, f"[Error] input shape must be 3D [B N D], got {len(input_shape)}D input: {input_shape}"

    # validate that the final dim of residual stream matches the intended embedding dimension
    assert_msg = f"[Error] Embedding dim must be last dimension and of size: {embed_dim}. Inputted tensor shape: {input_shape}."
    assert input_shape[-1] == embed_dim, assert_msg


# patch reshape tokenizer (ViT)
class PatchEmbedding(nn.Module):
    """Class that creates patch embeddings based on the ViT model (from paper '16 by 16 patches is all you need')
    Assumes a [B C H W] shaped input image, forward pass outputs a [B N D] tensor, where:
        B: batch size
        C: input channels
        H: image height in pixels
        W: image width in pixels
        N: transformer sequence length (N = H x W / P**2)
        D: hidden embedding dimension
    
    ARGS
        in_channels (int): number of channels for input image
        patch_size (int): size of each patch applied to the input image
        embed_dim (int): size of the hidden dimension of the model's embeddings
        embed_dropout (float): probability of elements being zeroed during training
    """

    def __init__(self,
                 in_channels:int=IN_CHANNELS,
                 patch_size:int=PATCH_SIZE,
                 embed_dim:int=EMBED_DIM,
                 embed_dropout:float=EMBED_DROPOUT,) -> None:
        super().__init__()
        
        # set attributes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.flattened_patch_dim = in_channels * patch_size**2

        # define mapping module
        self.embedding_map = nn.Sequential(
            nn.Linear(in_features=self.flattened_patch_dim,
                      out_features=embed_dim,
                      bias=False),
            nn.Dropout(p=embed_dropout)
        )
        
    def forward(self, input) -> torch.Tensor:
        # assertion that input image size is divisible by patch size and that x is 4d tensor
        vit_input_tensor_validation(input_tensor=input,
                                    patch_size=self.patch_size)

        # get key dimensions from inputs
        B, C, H, W = input.shape # extract dimensions from input
        P = self.patch_size
        N = int(H * W / P**2) # calculate sequence length (number of patches)

        # reshape input image and map to shape [batch_size x patch_sequence_length x embedding_dim]
        x = input.reshape(B, C, H // P, P, W // P, P) # prepare patch dimensions to avoid scrambling
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous() # ensures correct memory layout
        x = x.view(B, N, self.flattened_patch_dim) # view applies correct shape
        return self.embedding_map(x) # map flattened patches to embedding dimensions


# conv tokenizer (conv pool reshape)
class Tokenizer(nn.Module):
    """docstring"""
    # TODO tokenizer docstring

    def __init__(self,
                 out_channels:int,
                 in_channels:int=IN_CHANNELS,
                 kernel_size:int=KERNEL_SIZE,
                 stride:int=STRIDE,
                 padding:int=PADDING,
                 pooling_kernel:int=POOLING_KERNEL_SIZE,
                 pooling_stride:int=POOLING_STRIDE,
                 pooling_padding:int=POOLING_PADDING,) -> None:
        super().__init__()

        # TODO tokenizer init input checks

        # define tokenizer layers
        self.tokenizer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel,
                         stride=pooling_stride,
                         padding=pooling_padding,),
        )

        def forward(self, x) -> torch.Tensor:
            # TODO tokenizer fwd input checks

            # return applied tokenizer
            return self.tokenizer(x)


# TODO sine embeddings


# vit msa block
class MSABlock(nn.Module):
    """Single block of Multihead Self Attention, with Layer Norm applied before it.
    Assumes a [B N D] tensor input and provides the same dimensionality of output, where:
        B: batch size
        N: transformer sequence length (N = Height x Width / patch_size**2)
        D: hidden embedding dimension
    
    ARGS
        embed_dim (int): hidden dimension size for transformer
        num_heads (int): number of self-attention heads applied to hidden dimension
        attn_dropout (float): probability of attn weights being zeroed during training
    """

    def __init__(self,
                 embed_dim:int=EMBED_DIM,
                 num_heads:int=NUM_MSA_HEADS,
                 attn_dropout:float=ATTN_DROPOUT,) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.ln = nn.LayerNorm(normalized_shape=embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=attn_dropout,
                                         batch_first=True) # default order is [sequence, batch, feature]
        
    def forward(self, x) -> torch.Tensor:
        # validate shape and size of input tensor
        residual_stream_validation(residual_stream=x,
                                   embed_dim=self.embed_dim)

        # apply layer norm followed by MSA
        x = self.ln(x)
        attn_output, _ = self.msa(query=x,
                        key=x,
                        value=x,
                        need_weights=False)
        return attn_output


# vit mlp block
class MLPBlock(nn.Module):
    """Single Multi-Layer Perceptron block, with Layer Norm applied before it, dropout after each dense layer, and GELU non-linearity after hidden layer.
    Assumes a [B N D] tensor input and provides the same dimensionality of output, where:
        B: batch size
        N: transformer sequence length (N = Height x Width / patch_size**2)
        D: hidden embedding dimension
    
    ARGS
        embed_dim (int): hidden dimension size for transformer
        hidden_dim (int): dimensionality of the hidden MLP layer
        mlp_dropout (float): probability of an element being zeroed
    """

    def __init__(self,
                 embed_dim:int=EMBED_DIM,
                 hidden_dim:int=MLP_HIDDEN_DIM,
                 mlp_dropout:float=MLP_DROPOUT,) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.ln = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim,
                      out_features=hidden_dim,),
            nn.Dropout(p=mlp_dropout),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=embed_dim),
            nn.Dropout(p=mlp_dropout),
        )
    
    def forward(self, x) -> torch.Tensor:
        # validate shape and size of input tensor
        residual_stream_validation(residual_stream=x,
                                   embed_dim=self.embed_dim)

        # perform MLP block operations
        return self.mlp(self.ln(x))


# vit encoder block 
class ViTEncoderBlock(nn.Module):
    """A complete ViT encoder block, comprising of a MSA block followed by MLP block, each with skip-connections applied.
    Assumes a [B N D] tensor input and provides the same dimensionality of output, where:
        B: batch size
        N: transformer sequence length (N = Height x Width / patch_size**2)
        D: hidden embedding dimension
    
    ARGS
        embed_dim (int): hidden dimension size for transformer
        num_msa_heads (int): number of self-attention heads applied to hidden dimension
        attn_dropout (float): probability of an element in MSA being zeroed
        mlp_hidden_dim (int): dimensionality of the hidden MLP layer
        mlp_dropout (float): probability of an element in MLP being zeroed
        """

    def __init__(self,
                 embed_dim:int=EMBED_DIM,
                 num_msa_heads:int=NUM_MSA_HEADS,
                 attn_dropout:float=ATTN_DROPOUT,
                 mlp_hidden_dim=MLP_HIDDEN_DIM,
                 mlp_dropout:float=MLP_DROPOUT,) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.msa_block = MSABlock(embed_dim=embed_dim,
                                  num_heads=num_msa_heads,
                                  attn_dropout=attn_dropout,)
        self.mlp_block = MLPBlock(embed_dim=embed_dim,
                                  hidden_dim=mlp_hidden_dim,
                                  mlp_dropout=mlp_dropout)
        
    def forward(self, x) -> torch.Tensor:
        # validate shape and size of input tensor
        residual_stream_validation(residual_stream=x,
                                   embed_dim=self.embed_dim)
        
        # apply MSA block (note MSA optionally returns weights, so first element extracted) and add residual
        x = self.msa_block(x) + x

        # apply MLP block and add residual
        return self.mlp_block(x) + x


# sequence pooling
class SequencePool(nn.Module):
    """docstring"""
    # TODO sequence pool docstring

    def __init__(self,
                 embed_dim:int,) -> None:
        super().__init__()

        # TODO sequence pool init input checks

        # set attributes
        self.embed_dim = embed_dim

        # linear layer g(xL)
        self.linear = nn.Linear(in_features=embed_dim,
                                out_features=1,)
    
    def forward(self, x) -> torch.Tensor:
        # extract dims
        B, N, d = x.shape
       
       # TODO sequence pool fwd input checks
        assert_msg = f"[ERROR] Expected embed_dim={self.embed_dim}, x.shape[-1]={d}"
        assert d == self.embed_dim, assert_msg

       # xL_prime: softmax( g(xL).T ), [B N d] -> [B 1 N]
        x_prime = F.softmax(torch.transpose(self.linear(x), 1, 2),
                            dim=2)

        # xL_prime x xL: [B 1 N] x [B N d] -> [B 1 d]
        x = x_prime @ x

        # reshape to be [B d]
        return x.view(B, d)


# TODO cct


# TODO cvt


if __name__ == "__main__":
    xL = torch.randn(4, 8, 16)
    print(f"x0 shape: {xL.shape}")

    # apply SequencePool
    sequence_pool = SequencePool(embed_dim=16)
    z = sequence_pool(xL)
    print(f"z shape: {z.shape}")