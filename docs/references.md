# Research references

The repository links papers instead of checking PDF binaries into Git.

- Avazu dataset: [Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)
- Winning Avazu implementation: [ycjuan/kaggle-avazu](https://github.com/ycjuan/kaggle-avazu)
- Avazu-winning field-aware baseline: [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- Avazu top-five feature engineering: [Click-Through Rate Prediction: Top-5 Solution for the Avazu Contest](https://www.efimov-ml.com/pdfs/ClickThroughRate2015.pdf)
- DCNv2: [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank](https://arxiv.org/abs/2008.13535)
- SENet feature reweighting: [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433)
- Low-rank FiBiNET: [FiBiNet++: Reducing Model Size by Low Rank Feature Interaction Layer](https://arxiv.org/abs/2209.05016)
- Normalized representations: [nGPT: Normalized Transformer with Representation Learning on the Hypersphere](https://arxiv.org/abs/2410.01131)
- STEC: [See-Through Transformer-based Encoder for CTR Prediction](https://arxiv.org/abs/2308.15033)

The STEC implementation preserves the paper's defining pre-pooling Hadamard
interaction, multi-head attention path, stacked Add-and-Norm/FFN blocks,
`N + 1` interaction levels, per-level batch normalization, and concatenated
prediction path. Numerical fields use the paper's scalar-times-vector embedding
without an affine offset.

The nGPT implementation preserves hyperspherical hidden-state updates, matrix
and embedding reprojection after optimizer steps, normalized and rescaled
query/key attention, SwiGLU scaling, normalized output weights, learned logit
scaling, and the no-weight-decay/no-warmup optimizer recipe. CTR adaptation
replaces causal language tokens and RoPE with non-causal, namespace-specific
field tokens, a learned classification token, and two normalized class
prototypes; the normalized Transformer mechanics are unchanged.
