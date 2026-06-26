# Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting
PyTorch Implementation of TimeAlign.

## 📰 News

🚩 2026-01-26: TimeAlign has been accepted as ICLR 2026 Poster.

🚩 2025-09-21: Initial upload to arXiv ([PDF](https://arxiv.org/pdf/2509.14181)).

## 🌟 Overview

TimeAlign is a lightweight, plug-and-play framework that bridges the distributional gap in time series forecasting by aligning past and future representations through a reconstruction-based alignment task. The implementation of alignment is located in `./layers/Alignment.py`.

![](assets/pipeline.png)

## 🛠 Prerequisites

Ensure you are using Python 3.10.18 and install the necessary dependencies.

## 📊 Prepare Datastes

Begin by downloading the required datasets. All datasets are conveniently available at [iTransformer](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link). Create a separate folder named `./dataset` and neatly organize all the csv files as shown below:
```
dataset
└── electricity.csv
└── ETTh1.csv
└── ETTh2.csv
└── ETTm1.csv
└── ETTm2.csv
└── traffic.csv
└── weather.csv
└── solar_AL.txt
```

## 💻 Training

All scripts are located in `./scripts`. For instance, to train a model using the ETTh1 dataset with an input length of 720, simply run:

```shell
bash ./scripts/ETTh1.sh
```

After training:

- Your trained model will be safely stored in `./checkpoints`.
- Numerical results in .npy format can be found in `./results`.
- A comprehensive summary of quantitative metrics is accessible in `./result.txt`.

## 📚 Citation
If you find this repo useful, please consider citing our paper as follows:
```bibtex
@article{hu2025bridging,
  title={Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting},
  author={Hu, Yifan and Yang, Jie and Zhou, Tian and Liu, Peiyuan and Tang, Yujin and Jin, Rong and Sun, Liang},
  journal={arXiv preprint arXiv:2509.14181},
  year={2025}
}
```

## 🙏 Acknowledgement
Special thanks to the following repositories for their invaluable code and datasets:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [iTransformer](https://github.com/thuml/iTransformer)
- [DLinear](https://github.com/cure-lab/LTSF-Linear)
- [PatchTST](https://github.com/yuqinie98/PatchTST)

## 📩 Contact
If you have any questions, please contact [huyf0122@gmail.com](huyf0122@gmail.com) or submit an issue.
