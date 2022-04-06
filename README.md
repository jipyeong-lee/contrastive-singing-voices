# Self-Supervised Contrastive Learning for Singing Voices

This repository provides the implementation of our paper titled "Self-Supervised Contrastive Learning for Singing Voices," which is published in [IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)](https://doi.org/10.1109/TASLP.2022.3169627).

## Structure

This repository contains the following scripts:

- `train.py`: a script to run the self-supervised contrastive training
    - `--pitch` flag specifies whether to use pitch shifting for generating anchors
    - `--stretch` flag specifies whether to use time stretching for generating anchors
- `extract.py`: a script to extract feature representations using a model trained with `train.py`

## Citation

If you find our work helpful for your research, please consider citing the paper.

```
@article{yakura2022singingvoice,
    title   = {Self-Supervised Contrastive Learning for Singing Voices},
    author  = {Hiromu Yakura and Kento Watanabe and Masataka Goto},
    journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    year    = {2022},
    volume  = {30},
    pages   = {1614-1623},
    doi     = {10.1109/TASLP.2022.3169627}
}
```

## Notice

Part of the source code in this repository is inspired by [https://github.com/BestJuly/IIC](https://github.com/BestJuly/IIC).
