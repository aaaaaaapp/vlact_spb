# 方式 A：SFTP 上传 VLAct PI 权重

## 网关信息

| 项 | 值 |
|----|-----|
| Host | `106.75.235.248` |
| Port | `45432` |
| User | `ftpuser` |
| 远端目录 | `/upload` |
| 集群落盘 | `/mnt/jfzn/ftp_upload` |

**注意：** 公钥**只能**加在 `106.75.235.248` 上 **`ftpuser`** 的 `~/.ssh/authorized_keys`，不要加到集群 `liujn` 或其他账号。见同目录 `ftpuser_authorized_keys.snippet`。

## 1. 外部机准备文件（~10.5GB）

从 TOS **无 proxy** 拉取（源文件已验证完好）：

```bash
export TOSUTIL_DISABLE_PROXY=1
bash scripts/setup/external_download/01b_download_robotwin_ckpt_from_tos.sh
# 得到: $STAGING_MODELS/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt
```

校验：`stat -c%s` 应为 **11276441870**。

## 2. SFTP 上传

```bash
sftp -P 45432 -i /path/to/private_key ftpuser@106.75.235.248
sftp> cd /upload
sftp> put /path/to/steps_100000_pytorch_model.pt
# 或整目录:
sftp> put -r /path/to/VLAct_Qwen3PI_Robotwin_all_Finetune
sftp> bye
```

大文件上传约 10–30 分钟（视带宽）。

## 3. 集群自动导入（已部署 watcher）

```bash
# 已在后台监控；也可手动：
bash /mnt/jfzn/liujn/code/VLAct-main/scripts/setup/import_vlact_pi_from_ftp_upload.sh \
  /mnt/jfzn/ftp_upload/steps_100000_pytorch_model.pt
```

日志：`logs/setup/watch_ftp_upload_vlact_pi.log`
