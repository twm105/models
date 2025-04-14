# AI Models
<img src="images/nn_streets.png" alt_text="Neural net town image">
Repository of neural net models coded from tutorials, papers, or other ideas.

## Models
**CCT:** A Compact Convolutional Transformer (CCT) from the paper <a href="https://arxiv.org/pdf/2104.05704">Escaping the Big Data Paradigm with Compact Transformers</a>: a CNN/ViT hybrid that replaces the ViT CLS token with a 'Sequence Pool' layer, which combines all sequence tokens into a single embedding layer to classify on. A CVT variant uses ViT Patch Embeddings instead of the CNN tokeniser. Defaults to CCT-7/3x1 size.

**ConViT:** A hybrid CNN-ViT that uses inverted convolutional blocks (akin to MobileNet blocks, but without the depthwise Conv) followed by ViT blocks. Defaults to 50% CNN layers, 50% ViT layers of the base model size (i.e. 12 layers total).

**ViT:** A version of the original Vision Transformer from the paper <a href="https://arxiv.org/pdf/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a>. The ViT defaults to the base model size.