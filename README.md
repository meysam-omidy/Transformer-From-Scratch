
## Overview
This repository contains the implementation of transformer, inspired by the paper "Attention Is All You Need" by Vaswani et al ([link to the paper](https://arxiv.org/abs/1706.03762)). 
The Transformer is a pioneering neural network architecture that eschews recurrent and convolutional layers in favor of self-attention mechanisms. This model is renowned for its efficiency in parallelization and its superior performance on various machine translation tasks.

## Features

- **Pure Attention-Based Architecture**: Completely removes recurrence and convolutions.
- **Highly Parallelizable**: Allows faster training times compared to traditional models.
- **Configurable Hyperparameters**: Easily adjust layers, attention heads, and other settings.
- **Regularization Techniques**: Includes dropout and label smoothing for improved generalization.
- **Comprehensive Evaluation**: Tools to measure performance metrics such as BLEU score.

## Requirements
To run the code, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- Cuda (preferred to train the model faster)

## Sequential Forward Function
This function defined in the transformer class, is used to not only use the target tokens for generating output, but also the generated tokens of the model itself. It should be considered that sequential forward function is used during training. 

## Example Usage
Multihead self attention:
```
d1 = torch.randn(10, 5, 256) # batch_size=10, num_tokens=5, embedding_size=256
d2 = torch.randn(10, 3, 256) # batch_size=10, num_tokens=3, embedding_size=256
mha = MultiHeadAttention(main_dim=256, num_heads=4, k_dim=512, v_dim=1024)
mha(d1, d2).shape
$ (10, 3, 256)
```
Transformer:
```
d1 = torch.randint(0, 400, (100, 200)) # batch_size=100, num_tokens=200
d2 = torch.randint(0, 400, (100, 1000)) # batch_size=100, num_tokens=1000
transformer = Transformer(num_input_tokens=400, miain_dim=256, ff_dim=512, num_heads=4, num_encoder_layers=10, num_decoder_layers=10, num_output_tokens=500, max_tokens=1000)
transformer(d1,d2).shape
$ (100, 1000, 500)
transformer.sequential_forward(d1, d2, ratio=0.5).shape
$ (100, 1000, 500)
```
## References

```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
