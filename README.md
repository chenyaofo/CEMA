# Towards Robust and Efficient Cloud-Edge Elastic Model Adaptation via Selective Entropy Distillation

This is the official project repository for [Towards Robust and Efficient Cloud-Edge Elastic Model Adaptation via Selective Entropy Distillation](https://openreview.net/forum?id=vePdNU3u6n) [ICLR 2024], Authors: Yaofo Chen, Shuaicheng Niu, Yaowei Wang, Shoukai Xu, Hengjie Song, Mingkui Tan

**Dependencies Installation**:

Please refer to `requirements.txt`

**Data Preparation**:

Please download [ImageNet-C](https://github.com/hendrycks/robustness) and extract the tar files. Then use `tools/make_tfrec.py` to transfer raw image files into tfrecord format.

## Example: ImageNet-C Experiments

```
python -m entry.run --conf conf/cema.conf -o outputs/
```

## Citation

```
@inproceedings{chen2024towards,
  title={Towards Robust and Efficient Cloud-Edge Elastic Model Adaptation via Selective Entropy Distillation},
  author={Yaofo Chen and Shuaicheng Niu and Shoukai Xu and Hengjie Song and Yaowei Wang and Mingkui Tan},
  booktitle={International Conference on Learning Representations},
  year={2024},
}
```