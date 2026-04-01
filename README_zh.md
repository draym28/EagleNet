# EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval

[English](./README.md) | 中文

我们的论文被 **CVPR 2026** 接收。作者：[Yuhan Chen](https://github.com/draym28), [Pengwen Dai](https://scst.sysu.edu.cn/teacher/DaiPengwen), [Chuan Wang](https://chuanwang-cv.github.io/), [Dayan Wu](https://wudayan92.github.io/), [Xiaochun Cao](https://scst.sysu.edu.cn/teacher/CaoXiaochun)。

论文链接：[[arXiv]](https://arxiv.org/abs/2603.25267v2)。

## 简介

EagleNet 可准确地丰富文本嵌入的信息，减少与视频嵌入的语义差距，并使其具备感知视频上下文信息的能力：它同时建模视频中的“文本-帧交互”，以及视频内部的“帧-帧交互”。

- 细粒度关系学习（Fine-Grained Relationship Learning, FRL）：首先基于**生成的候选文本**和**帧**构建**文本-帧图**；随后学习文本、帧之间的复杂关系，用于最后将候选文本聚合为包含帧上下文信息的丰富文本嵌入。
- 能量感知匹配（Energy-Aware Matching, EAM）：用于刻画文本-帧交互的能量分布，从而更准确地捕捉真实的文本-视频对分布，进一步提升 FRL 中的细粒度关系学习效果。
- 我们将传统的基于 softmax 的对比损失替换为 sigmoid 损失，以获得更有效的跨模态对齐，并使得训练更加稳定。

EagleNet 在 MSRVTT（R@1 51.0%）、DiDeMo（51.5%）、MSVD（50.9%）以及 VATEX（63.6%）上取得了 SOTA 结果。

![framework](./image/framework.png)

## 代码结构

此目录包含一个基于 CLIP 的 **视频-文本检索（video-text retrieval）** 实现。
- 主要入口是 `main_my.py`。
- 训练/测试循环在 `train_and_eval.py` 中。
- 参数在 `args.py` 中定义。
- 数据集文件放在 `./data/`，最佳模型文件放在 `./best_ckpts/`。
- 为 MSR-VTT（9k）、DiDeMo、MSVD 和 VATEX 提供了训练脚本。

## 依赖要求

建议使用单独的 Conda（或 venv）环境。依赖集合与 [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) 风格的项目保持一致。

1. 创建 conda 环境：

```bash
conda create -n eaglenet python=3.8
conda activate eaglenet
```

2. 安装 pytorch：`torch=1.12.1`，`torchvision=0.13.1`。

3. 安装其他依赖：

```bash
ftfy
regex
tqdm
opencv-python
boto3
requests
pandas
```

如果你使用多 GPU 训练（分布式），请确保 `torch.distributed` 正常工作，并且你的 NCCL/驱动配置正确。

## 数据集

我们在 MSRVTT、DiDeMo、MSVD 和 VATEX 上测试模型。前三个数据集请参考 [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)，而 VATEX 请参考 [TS2-Net](https://github.com/LiuRicky/ts2_net)。

（可选）我们参考 [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) 将所有视频压缩到 3fps，并将分辨率设为 224*224 ，从而加速运行：

```bash
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```

此外，我们也提供通过 [Baidu Netdisk](https://pan.baidu.com/s/1OlrADWgLE9fWUdieB0JX_Q?pwd=qasj) 下载原始/压缩视频的链接。

划分/标注文件可以通过 [Google Driver](https://drive.google.com/drive/folders/1nqJfU2dYYJYNcT6X1yBCkhnj6Z0j4zHw?usp=drive_link) 或 [Baidu Netdisk](https://pan.baidu.com/s/1a6HoHzwp3UFJPhA3_Pk5Zg?pwd=ckch) 下载。

数据文件的组织结构如下：

```text
.
├── data/
│   ├── MSRVTT/
│   │   ├── msrvtt_data/
│   │   │   ├── MSRVTT_data.json
│   │   │   ├── MSRVTT_train.9k.csv
│   │   │   ├── MSRVTT_train.7k.csv
│   │   │   └── MSRVTT_JSFUSION_test.csv
│   │   ├── videos/
│   │   │   └── all/
│   │   │       └── *.mp4
│   │   └── ...
│   ├── DiDeMo/
│   │   ├── annotation/
│   │   │   └── (split/annotation files used by the DiDeMo dataloader)
│   │   └── videos/
│   │       └── *.mp4
│   ├── MSVD/
│   │   ├── msvd_data/
│   │   │   └── (split/annotation files used by the MSVD dataloader)
│   │   └── videos/
│   │       └── *.avi
│   └── VATEX/
│       ├── vatex_data/
│       │   └── (split/annotation files used by the VATEX dataloader)
│       ├── videos/
│       │   └── *.mp4
│       └── ...
└── ...
```

预训练的 CLIP 权重会自动下载。也可以手动下载：

- CLIP（ViT-B/32）权重

```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

- CLIP（ViT-B/16）权重

```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

然后将其放入 `~/.cache/clip/`。

## 如何运行

#### 从头训练

```bash
bash run_train_msrvtt9k-vit_b_32.sh
bash run_train_msrvtt9k-vit_b_16.sh
bash run_train_didemo.sh
bash run_train_msvd.sh
bash run_train_vatex.sh
```

#### 在 MSRVTT 上测试

MSRVTT（ViT-B/32 & ViT-B/16）的 EagleNet 检查点可以通过 [Google Driver](https://drive.google.com/drive/folders/1oy8nvyd6tbUqGVJZ44layuyaHSXcyQeP?usp=drive_link) 或 [Baidu Netdisk](https://pan.baidu.com/s/1Ps9iRkKOR9V8yQklGalY5Q?pwd=rief) 下载，然后运行：

```bash
bash run_eval_msrvtt9k-vit_b_32.sh
bash run_eval_msrvtt9k-vit_b_16.sh
```

## 致谢

感谢 [CLIP](https://github.com/openai/CLIP)、[CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)、[TS2-Net](https://github.com/LiuRicky/ts2_net) 以及 [XPool](https://github.com/layer6ai-labs/xpool) 的开发者开源其代码。

