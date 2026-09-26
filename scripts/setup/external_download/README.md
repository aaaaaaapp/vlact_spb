# 外部服务器下载脚本（VLAct 权重 + 预训练数据）

集群代理慢/不稳定时，在**有公网**的机器上下载，再同步到 `/mnt/jfzn`。

> **把脚本传到外部下载机**：见 [`EXTERNAL_SERVER_SETUP.md`](EXTERNAL_SERVER_SETUP.md)  
> 脚本包已上传 TOS：`tos://c20250405/ljn/transfer/vlact/toolkit/vlact_external_download_scripts.tar.gz`

## 1. 外部机器准备

```bash
# Python 3.10+ 环境
pip install -U "huggingface-hub>=0.23,<1.0" hf_transfer

# gated 数据集需要 Token（InternData-A1 等）
export HF_TOKEN="hf_xxxxxxxx"          # 或写入 ~/.huggingface/token
huggingface-cli login --token "$HF_TOKEN"

# 可选：国内机器用镜像
# export HF_ENDPOINT=https://hf-mirror.com

# 下载目录（默认 ~/vlact_external_download）
export STAGING_DIR="$HOME/vlact_external_download"
```

## 2. 下载

```bash
cd /path/to/VLAct-main/scripts/setup/external_download

# 只下 RobotWin 测评权重（~10.5GB，优先）
bash 01_download_robotwin_ckpt.sh

# 查看各数据集实际下载范围（多数为部分子目录，非全量 repo）
bash 02_download_one.sh --list

# 下全部预训练资源（按 datasets.conf.sh 清单，含 --include 部分下载）
bash 02_download_pretrain_all.sh

# 或按组 / 单个下载
bash 02_download_one.sh --group vlm          # LLaVA-COCO + ReCap-CC3M
bash 02_download_one.sh --group vla_core     # DROID + MolmoAct + RoboInter
bash 02_download_one.sh llava_recap_cc3m     # 单个

# InternData-A1 split_aloha 10 个 tar.gz（gated，~TB 级，按需）
bash 03_download_internedata.sh
# 或分片：INTERNDATA_SHARD=0 INTERNDATA_SHARDS=3 bash 03_download_internedata.sh
```

## 3. 传到集群（推荐：tosutil / TOS）

集群已有 tosutil，配置在 `/mnt/jfzn/liujn/.tosutilconfig`，二进制在  
`/mnt/jfzn/liujn/code/Starvla/Starvla/tosutil`。

### 3a. 外部机器准备 tosutil

```bash
# 从集群拷贝 tosutil 和配置（只需做一次）
scp liujn@10.120.154.139:/mnt/jfzn/liujn/code/Starvla/Starvla/tosutil ~/bin/
scp liujn@10.120.154.139:/mnt/jfzn/liujn/.tosutilconfig ~/
chmod +x ~/bin/tosutil

export TOSUTIL=~/bin/tosutil
export TOSUTIL_BASE_CONF=~/.tosutilconfig
```

### 3b. 外部：下载 → 上传 TOS

```bash
source 00_env.sh
bash 01_download_robotwin_ckpt.sh
bash 05_upload_to_tos.sh --only robotwin
# 预训练数据: bash 02_download_pretrain_all.sh && bash 05_upload_to_tos.sh --only pretrain
```

上传到：`tos://c20250405/ljn/transfer/vlact/`（可用 `TOS_BUCKET_PREFIX` 改）

### 3c. 集群：从 TOS 下载

```bash
cd /mnt/jfzn/liujn/code/VLAct-main/scripts/setup/external_download
bash 06_download_from_tos.sh --only robotwin
# 预训练: bash 06_download_from_tos.sh --only pretrain
```

支持断点续传，日志在 `/tmp/vlact-tos-transfer/`。

---

## 3-alt. 同步回集群（rsync，备选）

```bash
bash 04_sync_to_cluster.sh --dry-run
bash 04_sync_to_cluster.sh --execute --only robotwin
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `STAGING_DIR` | `~/vlact_external_download` | 外部下载根目录 |
| `CLUSTER_HOST` | `liujn@10.120.154.139` | SSH 目标 |
| `TOS_BUCKET_PREFIX` | `tos://c20250405/ljn/transfer/vlact` | TOS 路径 |

## 4. 下载清单与集群目标路径

**注意：预训练不需要各 HF repo 全量**，脚本已用 `--include` 限制范围（定义在 `datasets.conf.sh`）。

| 资源 | HF Repo | 实际下载范围 | 约大小 | 集群路径 |
|------|---------|--------------|--------|----------|
| **VLAct RobotWin PI** | `StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune` | `checkpoints/steps_100000_pytorch_model.pt` | ~11 GB | `models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/` |
| Qwen3-VL-4B-Instruct-Action | `StarVLA/Qwen3-VL-4B-Instruct-Action` | 全量 | ~8 GB | `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/` |
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | 全量 | ~1.5 GB | `playground/Pretrained_models/Qwen3-0.6B/` |
| LLaVA-OneVision-COCO | `StarVLA/LLaVA-OneVision-COCO` | `sharegpt4v_coco.zip`, `llava_jsons/**` | ~2.5 GB | `playground/Datasets/LLaVA-OneVision-COCO/` |
| LLaVA-ReCap-CC3M | `lmms-lab/LLaVA-ReCap-CC3M` | **`data/**` 仅 parquet**（非全量 139GB） | ~数 GB | `playground/Datasets/LLaVA-ReCap-CC3M/` |
| DROID | `cadene/droid` | 全量 | ~400 GB | `playground/Datasets/DROID/` |
| DROID_100 | `lerobot/droid_100` | 全量（**可选**） | ~0.4 GB | `playground/Datasets/DROID_100/` |
| MolmoAct | `allenai/MolmoAct-Dataset` | **`household/train` + `tabletop/train`** | 远小于全量 | `playground/Datasets/MolmoAct-Dataset/` |
| RoboInter-Data | `InternRobotics/RoboInter-Data` | `Annotation_with_action_lerobotv21/lerobot_droid_anno/**` | ~数十 GB | `playground/Datasets/RoboInter-Data/` |
| InternData-A1 | `InternRobotics/InternData-A1` | **10 个 split_aloha tar.gz**（见 `03_`） | ~TB | `playground/Datasets/InternData-A1_raw/` |
| RoboCOIN ×10 | `RoboCOIN/<task>` | 各 task repo 全量 | 各若干 GB | `playground/Datasets/RoboCOIN/<task>/` |

DROID_100 若已下载 RoboInter 标注子集，集群 `postprocess_datasets.sh` 可自动从 RoboInter 生成，不必单独下 `lerobot/droid_100`。

## 5. 同步完成后（集群上）

```bash
cd /mnt/jfzn/liujn/code/VLAct-main
conda activate vlact

# RobotWin smoke（权重同步后）
bash scripts/setup/run_robotwin_smoke_vlact_pi.sh

# 预训练数据后处理
bash scripts/setup/postprocess_datasets.sh
bash scripts/setup/build_pretrain_caches.sh
```
