# 外部下载服务器 — 快速部署

在**有公网**的机器上下载 VLAct 数据，再通过 TOS 传回集群。

## 第一步：把脚本传到外部服务器

### 方式 A — scp（推荐，最简单）

在**集群**上打包：

```bash
bash /mnt/jfzn/liujn/code/VLAct-main/scripts/setup/external_download/pack_for_external_server.sh
```

传到外部机（把 `USER@HOST` 换成你的外部服务器）：

```bash
scp /tmp/vlact_external_download_scripts.tar.gz USER@HOST:~/
ssh USER@HOST 'mkdir -p ~/vlact_external_download && tar xzf ~/vlact_external_download_scripts.tar.gz -C ~/vlact_external_download --strip-components=0'
```

解压后目录：`~/vlact_external_download/external_download/`

### 方式 B — 经 TOS 中转

**集群**上传：

```bash
bash /mnt/jfzn/liujn/code/VLAct-main/scripts/setup/external_download/08_upload_scripts_to_tos.sh
```

**外部机**拉取（需先有 tosutil + 配置，可从集群 scp）：

```bash
scp liujn@10.120.154.139:/mnt/jfzn/liujn/code/Starvla/Starvla/tosutil ~/bin/
scp liujn@10.120.154.139:/mnt/jfzn/liujn/.tosutilconfig ~/
chmod +x ~/bin/tosutil

mkdir -p ~/vlact_external_download && cd ~/vlact_external_download
~/bin/tosutil cp tos://c20250405/ljn/transfer/vlact/toolkit/vlact_external_download_scripts.tar.gz ./ \
  -conf=~/.tosutilconfig
tar xzf vlact_external_download_scripts.tar.gz
```

---

## 第二步：外部机环境

```bash
cd ~/vlact_external_download/external_download

pip install -U "huggingface-hub>=0.23,<1.0" hf_transfer

# gated 数据集需要
export HF_TOKEN="hf_xxxx"   # 或 huggingface-cli login
export STAGING_DIR="$HOME/vlact_external_download/data"

source 00_env.sh
bash 02_download_one.sh --list    # 查看各数据集实际下载范围
```

---

## 第三步：下载

```bash
# 优先：RobotWin 测评权重 ~11GB
bash 01_download_robotwin_ckpt.sh

# 预训练（按组，含部分下载 include）
bash 02_download_one.sh --group models
bash 02_download_one.sh --group vlm
bash 02_download_one.sh --group vla_core
bash 02_download_one.sh --group robocoin

# InternData split_aloha（gated，按需）
bash 03_download_internedata.sh
```

---

## 第四步：传回集群（TOS）

外部机需 tosutil（见方式 B 前两行 scp）：

```bash
cd ~/vlact_external_download/external_download
source 00_env.sh
export TOSUTIL=~/bin/tosutil
export TOSUTIL_BASE_CONF=~/.tosutilconfig

bash 05_upload_to_tos.sh --only robotwin    # 或 --only pretrain / --only all
```

**集群**拉取：

```bash
cd /mnt/jfzn/liujn/code/VLAct-main/scripts/setup/external_download
bash 06_download_from_tos.sh --only robotwin
```
