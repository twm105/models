import torch
from torch import nn
from typing import Tuple


# default config
BATCH_SIZE = 32
IN_CHANNELS = 3
IMG_SIZE = 224
PATCH_SIZE = 16
EMBED_TYPE = 'default'
EMBED_DIM = 768
EMBED_DROPOUT = 0
NUM_MSA_HEADS = 12
ATTN_DROPOUT = 0
MLP_HIDDEN_DIM = 3072
MLP_DROPOUT = 0.1
NUM_LAYERS = 12
NUM_CLASSES = 1000
CLASSIFIER_DROPOUT = 0.1


def input_tensor_validation(input_tensor:torch.Tensor,
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
                 embed_dropout:float=EMBED_DROPOUT,) -> nn.Module:
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
        input_tensor_validation(input_tensor=input,
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


class PatchEmbeddingCNN(nn.Module):
    """Class that creates patch embeddings based on the hybrid version of the ViT model (from paper '16 by 16 patches is all you need')
    Assumes a [B C H W] shaped input image, forward pass outputs a [B N D] tensor, where:
        B: batch size
        C: input channels
        H: image height in pixels
        W: image width in pixels
        N: transformer sequence length (N = Height x Width / patch_size**2)
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
                 embed_dropout:float=EMBED_DROPOUT,) -> nn.Module:
        super().__init__()
        
        self.patch_size = patch_size

        self.cnn_embed_map = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=embed_dim,
                      kernel_size=patch_size,
                      stride=patch_size,
                      padding=0,
                      bias=False,),
            nn.Flatten(start_dim=2,
                       end_dim=-1,),
            nn.Dropout(p=embed_dropout,)
        )
        
    def forward(self, x) -> torch.Tensor:
        # assertion that input image size is divisible by patch size and that x is 4d tensor
        input_tensor_validation(input_tensor=x,
                                patch_size=self.patch_size)
        
        # apply cnn, flatten and permute dims to correct shape
        return self.cnn_embed_map(x).permute(0, 2, 1)
    

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
                 attn_dropout:float=ATTN_DROPOUT,) -> nn.Module:
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
                 mlp_dropout:float=MLP_DROPOUT,) -> nn.Module:
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
                 mlp_dropout:float=MLP_DROPOUT,) -> nn.Module:
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
    

class ViT(nn.Module):
    """Vision Transformer class from the paper '16x16 patches are all you need.
    Assumes a [B C H W] shaped input image, contains a [B N D] residual stream, and outputs [B Y], where:
        B: batch size
        C: input channels
        H: image height in pixels
        W: image width in pixels
        N: transformer sequence length (N = H x W / P**2)
        D: hidden embedding dimension
        Y: logits for classifier
    
    ARGS
        img_shape (Tuple): 2d size of the input images accepted by the ViT in pixels
        in_channels (int): number of channels for input image
        patch_size (int): size of each patch applied to the input image
        embed_type (str): type of patch embeddings used: reshap vs cnn (hybrid)
        embed_dim (int): size of the hidden dimension of the model's embeddings
        embed_dropout (float): probability of an element being zeroed
        num_msa_heads (int): number of self-attention heads applied to hidden dimension
        attn_dropout (float): probability of an element being zeroed
        mlp_hidden_dim (int): dimensionality of the hidden MLP layer
        mlp_dropout (float): probability of an element being zeroed
        num_layers (int): the depth of ViT encoder blocks stacked together
        num_classes (int): the number of classes (logits) outputted by the model
        classifier_dropout (float): probability of an element being zeroed
    """

    def __init__(self,
                 img_shape:Tuple[int, int],
                 in_channels:int=IN_CHANNELS,
                 patch_size:int=PATCH_SIZE,
                 embed_type:str=EMBED_TYPE,
                 embed_dim:int=EMBED_DIM,
                 embed_dropout:float=EMBED_DROPOUT,
                 num_msa_heads:int=NUM_MSA_HEADS,
                 attn_dropout:float=ATTN_DROPOUT,
                 mlp_hidden_dim:int=MLP_HIDDEN_DIM,
                 mlp_dropout:float=MLP_DROPOUT,
                 num_layers:int=NUM_LAYERS,
                 num_classes:int=NUM_CLASSES,
                 classifier_dropout:float=CLASSIFIER_DROPOUT) -> nn.Module:
        super().__init__()

        # check inputs
        assert len(img_shape) == 2, f"2d image shape expected, {len(img_shape)}d provided."
        input_tensor_validation(input_tensor=torch.Tensor(1, in_channels, *img_shape),
                                patch_size=patch_size)

        # set attributes
        self.N = img_shape[0] * img_shape[1] // patch_size**2
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # define the type of embeddings
        if embed_type == 'default':
            self.patch_embeddings = PatchEmbedding(in_channels=in_channels,
                                                   patch_size=patch_size,
                                                   embed_dim=embed_dim,
                                                   embed_dropout=embed_dropout,)
        elif embed_type == 'hybrid':
            self.patch_embeddings = PatchEmbeddingCNN(in_channels=in_channels,
                                                      patch_size=patch_size,
                                                      embed_dim=embed_dim,
                                                      embed_dropout=embed_dropout,)
        else:
            raise ValueError(f"[Error] Unrecognised embedding type '{embed_type}'. Embedding type must be one of 'default' or 'hybrid'.")
        
        # create class token (expand batch size during forward pass)
        self.class_token = nn.Parameter(data=torch.randn(1, 1, embed_dim))

        # create position embeddings (expand batch size and sequence length (num_patches + class token) during forward pass)
        self.position_embeddings = nn.Parameter(data=torch.randn(1, self.N + 1, embed_dim))
        
        # define num_layers of the ViT encoder block
        self.encoder_blocks = nn.Sequential(
            *[ViTEncoderBlock(embed_dim=embed_dim,
                              num_msa_heads=num_msa_heads,
                              attn_dropout=attn_dropout,
                              mlp_hidden_dim=mlp_hidden_dim,
                              mlp_dropout=mlp_dropout) for _ in range(num_layers)]
        )
        
        # define the classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(in_features=embed_dim,
                      out_features=num_classes,),
        )
    
    def forward(self, x) -> torch.Tensor:
        # apply patch embeddings
        x = self.patch_embeddings(x)

        # expand class token along batch dimension
        class_token = self.class_token.expand(x.shape[0], -1, -1)

        # concatenate class token and patch embeddings on sequence dimension
        x = torch.cat([class_token, x], dim=1)

        # add expanded position embeddings (both batch size and sequence length) to class and patch embeddings
        x = x + self.position_embeddings # broadcasts position_embeddings along the batch dimension based on x input

        # apply num_layers ViT encoder blocks
        x = self.encoder_blocks(x)

        # apply classifier to the class token's embeddings only (0th element of sequence dimension)
        return self.classifier(x[:, 0, :].reshape(-1, x.shape[-1])) # can't use squeeze in case batch size is 1


if __name__ == '__main__':
    
    # config
    batch_size = BATCH_SIZE
    in_channels = IN_CHANNELS
    img_size = IMG_SIZE
    patch_size = PATCH_SIZE
    embed_type = EMBED_TYPE
    embed_dim = EMBED_DIM
    embed_dropout = EMBED_DROPOUT
    num_msa_heads = NUM_MSA_HEADS
    attn_dropout = ATTN_DROPOUT
    mlp_hidden_dim = MLP_HIDDEN_DIM
    mlp_dropout = MLP_DROPOUT
    num_layers = NUM_LAYERS
    num_classes = NUM_CLASSES
    classifier_dropout = CLASSIFIER_DROPOUT

    # create dummy image
    img = torch.ones([batch_size, in_channels, img_size, img_size])
    print(f"Input image shape: {img.shape}")
    
    # create default embeddings
    patch_embed_default = PatchEmbedding(in_channels=in_channels,
                                         patch_size=patch_size,
                                         embed_dim=embed_dim,
                                         embed_dropout=embed_dropout,)
    patch_embeddings_default = patch_embed_default(img)
    print(f"Default patch embeddings shape: {patch_embeddings_default.shape}")
    
    # create hybrid (CNN) embeddings
    patch_embed_CNN = PatchEmbeddingCNN(in_channels=in_channels,
                                        patch_size=patch_size,
                                        embed_dim=embed_dim,
                                        embed_dropout=embed_dropout,)
    patch_embeddings_hybrid = patch_embed_default(img)
    print(f"Hybrid patch embeddings shape: {patch_embeddings_hybrid.shape}")

    # set embeddings type for subsequent operations
    if embed_type == 'default':
        patch_embeddings = patch_embeddings_default
    elif embed_type == 'hybrid':
        patch_embeddings = patch_embeddings_hybrid
    else:
        raise ValueError(f"[Error] Unrecognised embedding type '{embed_type}'. Embedding type must be one of 'default' or 'hybrid'.")
    print(f"Patch embeddings mode: {embed_type}")

    # add class token and positional embeddings for purpose of debug
    class_token = torch.randn((batch_size, 1, embed_dim))
    patch_embeddings = torch.cat([class_token, patch_embeddings], dim=1)
    position_embeddings = torch.randn(patch_embeddings.shape)
    print(f"Class token and patch embeddings shape: {patch_embeddings.shape}")
    print(f"Position embeddings shape: {position_embeddings.shape}")
    patch_embeddings = patch_embeddings + position_embeddings

    # apply multihead self attention to residual stream
    msa = MSABlock(embed_dim=embed_dim,
                   num_heads=num_msa_heads,
                   attn_dropout=attn_dropout,)
    msa_out = msa(patch_embeddings)
    print(f"MSA output shape: {msa_out.shape}")

    # apply multi-layer perceptron to residual stream
    mlp = MLPBlock(embed_dim=embed_dim,
                   hidden_dim=mlp_hidden_dim,
                   mlp_dropout=mlp_dropout,)
    mlp_out = mlp(msa_out)
    print(f"MLP output shape: {mlp_out.shape}")

    # apply full ViT encoder block to residual stream
    vit_encoder_block = ViTEncoderBlock(embed_dim=embed_dim,
                                        num_msa_heads=num_msa_heads,
                                        attn_dropout=attn_dropout,
                                        mlp_hidden_dim=mlp_hidden_dim,
                                        mlp_dropout=mlp_dropout,)
    vit_encoder_block_out = vit_encoder_block(patch_embeddings)
    print(f"ViT encoder block output shape: {vit_encoder_block_out.shape}")

    # apply full ViT to input image
    vit = ViT(img_shape=(img_size, img_size),
              in_channels=in_channels,
              patch_size=patch_size,
              embed_type=embed_type,
              embed_dropout=embed_dropout,
              embed_dim=embed_dim,
              num_msa_heads=num_msa_heads,
              attn_dropout=attn_dropout,
              mlp_hidden_dim=mlp_hidden_dim,
              mlp_dropout=mlp_dropout,
              num_layers=num_layers,
              num_classes=num_classes,
              classifier_dropout=classifier_dropout)
    logits = vit(img)
    print(50 * '-')
    print(f"ViT logits shape: {logits.shape}")
    print(50 * '-')