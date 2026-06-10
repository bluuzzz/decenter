# DeCenter: Density-Guided Object Center Heatmap for Tiny Object Detection

## Installation
```sh
conda create -n decenter python=3.9 -y
conda activate decenter
bash install.sh
```

## Training
* Changed the paths in DeCenter.sh
* Pretrained weights can be found in the DQ-DETR repository.
```sh
bash scripts/DeCenter.sh
```

## Eval models
* Changed the paths in DeCenter_eval.sh
```sh
bash scripts/DeCenter_eval.sh
```

## Citation
```bibtex
@article{shi2026decenter,
  title={DeCenter: Density-Center Guided Perception Enhancement for UAV Object Detection},
  author={Shi, Zhiqing and Wu, Zhihao and Wen, Jie and Li, Mu and Fan, Xiaopeng and Wang, Yaowei and Shen, Linlin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

## Reference
This project is built upon [DQ-DETR](https://github.com/hoiliu-0801/DQ-DETR). 

```bibtex
@article{huang2024dq,
  title={Dq-detr: Detr with dynamic query for tiny object detection},
  author={Huang, Yi-Xin and Liu, Hou-I and Shuai, Hong-Han and Cheng, Wen-Huang},
  journal={arXiv preprint arXiv:2404.03507},
  year={2024}
}
```
