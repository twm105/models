# AI Models
<img src="images/nn_streets.png" alt_text="Neural net town image">
Repository of neural net models coded from tutorials, papers, or other ideas.

## Models
**CCT:** A Compact Convolutional Transformer (CCT) from the paper <a href="https://arxiv.org/pdf/2104.05704">Escaping the Big Data Paradigm with Compact Transformers</a>: a CNN/ViT hybrid that replaces the ViT CLS token with a 'Sequence Pool' layer, which combines all image sequence-tokens' features into a single embedding vector to classify. These provide improved model dimensionality and parameter utilisation for classification. A CVT variant uses ViT Patch Embeddings instead of the CNN tokeniser. Defaults to CCT-7/3x1 size. Trained with paper's training recipe on CIFAR-10 (incl. randomaug, mixup, cutmix) using A100 on Google Colab, then experimented with different training recipes to further drive-up test-accuracy. Achieved 94.61% accuracy with 3.76m parameter model over 300 epochs.
<img src="https://github.com/twm105/models/blob/main/images/250504-cct-7_3x1-3p76m-allmixup-CIFAR10-94p61acc-300epochs.png" alt_text="Training CCT on CIFAR-10">

**DeepSeek v3 MoE (WIP):** Implement the MoE architecture from DeepSeekv3's MoE (December 2024), including top-k expert routing/gating and expert-balancing buffer. Initial architecture developed using PyTorch in March 2025 but not trained.

**ConViT:** A hybrid CNN-ViT that uses inverted convolutional blocks (akin to MobileNet blocks, but without the depthwise Conv) followed by ViT blocks. Defaults to 50% CNN layers, 50% ViT layers of the base model size (i.e. 12 layers total).

**ViT:** A version of the original Vision Transformer from the paper <a href="https://arxiv.org/pdf/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a>. The ViT defaults to the base model size.
