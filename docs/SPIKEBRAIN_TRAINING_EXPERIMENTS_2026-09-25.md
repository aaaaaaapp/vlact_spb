# SpikeBrain-VLAct 训练稳定性与速度实验记录

日期：2026-09-25

本文记录本轮针对 SpikeBrain-VLAct 预训练做过的实际修改、ACP 实验、失败原因和当前结论。所有速度结论都来自真实日志，不把未完成或不同口径的实验混在一起。

## 1. 目标与固定约束

本轮目标是在不改变训练语义的前提下，提高 SpikeBrain-VLAct 的训练速度，并验证能否关闭 gradient checkpointing（下文简称 GC）。

固定训练约束如下：

- VLM 是 SpikeBrain-2.0-VL；Qwen3-0.6B 只作为 PI action expert。
- action heads 保留 `OFT + GR00T + PI`。
- VLM learning rate：`1e-5`。
- action learning rate：`1e-4`。
- weight decay：`0.01`。
- 全程 BF16，不引入 FP32 GDN 或 attention RMSNorm。
- `pack=false`。
- `camera_bucket_sampling=false`。
- 图像分辨率由 VLAct 数据流程控制，不使用 SpikeBrain-VLM 固定分辨率。
- 数据使用 `/mnt/jfzn/liujn/code/VLAct-main/playground/Datasets`。
- W&B 使用 offline 模式。
- W&B/训练日志记录 VLM、action 和 global 三类 gradient norm。

## 2. 已落地的代码修改

### 2.1 去掉 GDN 每层的 GPU 到 CPU 同步

文件：

`/mnt/jfzn/liujn/code/Starvla/Starvla/spikebrain/SpikeBrain-2.0-VL/sse_swa.py`

原实现通过 `lengths.max().item()` 计算 radix sort 的位数，这会在每层触发 GPU 到 CPU 的同步。由于这一处的最大长度就是当前序列长度 `L`，改为：

```python
bits_pos = int(L).bit_length()
```

排序语义不变，但去掉了该同步点。

### 2.2 降低 hidden-state finite 检查频率

文件：

`starVLA/model/modules/vlm/SpikeBrain2VL.py`

策略：

- 前 3 次 forward 必查，尽早发现 BF16 数值问题。
- 之后由 `framework.qwenvl.nonfinite_check_interval` 控制。
- 正式 launcher 默认每 100 次检查一次。

这样保留 fail-fast 能力，但不在每个 forward 都做全 tensor finite reduction。

### 2.3 gradient norm 只在真正记录日志的 step 计算

文件：

`starVLA/training/train_starvla_cotrain.py`

新增 `_should_collect_gradient_norms()`，只在 optimizer step 且该 step 满足 `logging_frequency` 时统计：

- `grad_norm/qwen_vl_interface`
- `grad_norm/action_model`
- `grad_norm/global`

30-step 探针在 step 10、20、30 都确认记录了三项指标。

### 2.4 kernel cache 改为“共享盘持久化、本地盘运行”

文件：

`scripts/run_scripts/Pretrain/pretrain_spikebrain_single_node.sh`

当前流程：

1. ACP 启动时，从仓库的 `.kernel_cache` 复制已有 cache 到节点本地 `/tmp`。
2. 训练期的 Triton、TorchInductor、Torch extensions 和 XDG cache 全部指向 `/tmp`。
3. 训练结束后，将新增 cache 写回共享盘，供后续任务预热。

共享持久 cache 当前约 `1.2G`、`34612` 个文件。实验表明，cache 内容本身有明显收益，但训练期直接在 AFS 上查询数万个小文件会产生严重额外延迟。

### 2.5 其他 launcher 改动

文件：

`scripts/run_scripts/Pretrain/pretrain_spikebrain_single_node.sh`

- VLM LR 默认值修正为 `1e-5`。
- action LR 默认值为 `1e-4`。
- weight decay 默认值为 `0.01`。
- `ENABLE_GC` 可由环境变量控制，安全默认值为 `true`。
- 支持 `NONFINITE_CHECK_INTERVAL` 和 `ACCELERATE_CONFIG_FILE`。
- 数据目录默认指向 VLAct-main 的数据目录。

新增或调整的探针入口：

- `scripts/setup/run_acp_spikebrain_nogc_probe.sh`
- `scripts/setup/run_acp_spikebrain_gc_cache_probe.sh`
- `starVLA/config/deepseeds/ds_config_small_bucket.yaml`
- `starVLA/config/deepseeds/deepspeed_zero2_small_bucket.yaml`

## 3. 关闭 GC 的实际实验

所有实验均为 2 x A100-80GB。结论不是估算，而是来自实际 OOM。

| 实验 | VLA BS/GPU | VLM BS/GPU | GAS | ZeRO bucket | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| no-GC v2 | 16 | 2 | 1 | 500M | 第一轮 action forward OOM |
| no-GC v3 | 8 | 1 | 2 | 500M | 完成少量 optimizer step 后，在 ZeRO-2 backward 初始化约 954 MiB buffer 时 OOM |
| no-GC v4 | 8 | 1 | 2 | 50M | 仍在 action backward 的 ZeRO-2 gradient partition buffer 分配时 OOM |

### 3.1 no-GC v2：原 batch

任务使用原始每卡 batch：VLA 16、VLM 2、GAS 1。

OOM 时：

- 单卡总显存约 79.32 GiB。
- GPU 0 已使用约 79.25 GiB，只剩约 51 MiB。
- 最后一次仅尝试再申请 72 MiB 就失败。

结论：原 batch 下关闭 GC 明确不可用。

### 3.2 no-GC v3：batch 减半并用 GAS=2 保持等效 global batch

配置：VLA 8、VLM 1、GAS 2。

部分 micro-step 时间：`181.70s, 83.30s, 82.21s, 68.43s, 33.62s`。

随后 ZeRO-2 backward 尝试申请约 954 MiB 时 OOM，当时 GPU 只剩约 571 MiB。

结论：减小 activation batch 仍不能消除 ZeRO-2 backward 的固定显存需求。

### 3.3 no-GC v4：继续缩小 ZeRO bucket

配置仍为 VLA 8、VLM 1、GAS 2，将 reduce/allgather bucket 从 500M 降到 50M。

前三个 micro-step 为 `41.73s, 21.97s, 19.61s`，说明预热 cache 已生效；但随后尝试分配 3.67 GiB gradient partition buffer 时 OOM，当时只剩 2.34 GiB。

结论：缩小通信 bucket 不能消除这一固定梯度分区缓冲。继续缩 batch 也没有足够依据保证稳定，因此正式 100k 训练保留 GC。

## 4. GC + cache 对照实验

三组日志使用相同的前 30 个输入 step，因此快慢 step 的形状分布基本逐项对应，适合做 paired comparison。

| 配置 | 前 30 步 mean | median | p90 | step 21-30 mean |
| --- | ---: | ---: | ---: | ---: |
| 旧基线，无持久 cache | 52.401s | 55.909s | 71.700s | 35.207s |
| GC + 运行期直接访问 AFS cache | 54.508s | 62.271s | 84.405s | 38.978s |
| GC + `/tmp` 本地运行 cache | **16.154s** | **17.310s** | **22.009s** | **11.473s** |

本地 cache 相对旧基线：

- 按 30 步总时间计算，约 `3.24x` 加速。
- 按逐 step 加速比取平均，约 `2.98x`。
- 相对直接访问 AFS cache，按均值约 `3.37x` 加速。

### 4.1 共享 AFS cache 探针 v1

显示名：`vlact-spikebrain-gc-cache-2gpu-30step-v1`

日志：

`logs/setup/acp_vlact_spikebrain_gc_cache_2gpu_30step_v1.log`

结果：

- 30 个训练 step 全部完成，最终模型和 offline W&B 数据均已保存。
- step 10/20/30 均记录了三类 grad norm。
- 首步从旧基线的 210.66s 降到 96.98s，但后续多数同 shape step 比旧基线更慢。
- 原因是训练期直接在 AFS 上频繁查询约 3.4 万个小 cache 文件。

任务最后被 ACP 标为 FAILED，但不是训练 OOM。训练完成后，运行中的 bash launcher 恰好遇到 launcher 文件被在线修改，出现：

```text
line 183: l_num_batches}: command not found
```

因此训练产物有效，但 shell 返回失败，cache 写回也报 warning。该失败 ACP 记录已经删除。之后不再在任务运行过程中修改其 launcher，v2 完整成功，验证了当前文件本身语法和收尾逻辑正常。

### 4.2 节点本地 cache 探针 v2

显示名：`vlact-spikebrain-gc-localcache-2gpu-30step-v2`

资源名：`pt-gup00poh`

ACP 状态：`SUCCEEDED`

日志：

`logs/setup/acp_vlact_spikebrain_gc_localcache_2gpu_30step_v2.log`

关键数据：

- step 1：43.677s
- step 2：22.911s
- step 3：21.750s
- step 21：4.480s
- step 26：3.833s
- step 30：17.934s
- 30-step 训练部分总耗时约 8 分 27 秒。
- 没有 OOM 或 traceback。
- 最终模型保存成功。
- W&B offline 记录成功。
- step 10/20/30 的 VLM/action/global grad norm 全部存在。

step 30 的 gradient norm 示例：

```text
grad_norm/qwen_vl_interface = 170.22055053710938
grad_norm/action_model      = 709.8936767578125
grad_norm/global            = 730.0164848315976
```

### 4.3 SpikeBrain 专用 Liger 数值验证与速度 A/B

新增文件：

- `starVLA/model/modules/vlm/spikebrain_liger.py`
- `scripts/setup/validate_spikebrain_liger.py`
- `scripts/setup/run_acp_spikebrain_liger_validation.sh`

SpikeBrain 的 MRoPE 不是 Qwen 的 position-id/frequency 生成方式。适配实现不复用 Qwen
MRoPE，而是直接消费 SpikeBrain 已生成的 interleaved T/H/W cos/sin，并分别处理：

- 标准 attention 的 `[B,H,S,D]` 布局。
- SpikeBrain 自定义 attention 的 `[B,S,H,D]` 布局。
- 视觉 RoPE 保持原生实现，不做 patch。

完整 6.6B 模型在 BF16 下先做 native-to-native 重复性对照，再原地启用单项 Liger，
比较 loss、最终 hidden state、代表参数梯度样本和完整梯度范数。native-to-native 的所有
差异均为 0，排除了模型自身非确定性。

| 配置 | loss relative | hidden relative L2 | 最大 gradient relative L2 | 结论 |
| --- | ---: | ---: | ---: | --- |
| RoPE only | **0** | **0** | **0** | 严格一致，可用 |
| RoPE + RMSNorm | 0.412% | 2.872% | 80.0% | 禁用 RMSNorm |
| RoPE + SwiGLU | **0** | **0** | 3.568% | backward 不一致，禁用 SwiGLU |
| fused linear CE 微测 | 0.169% | N/A | 约 0.3% | SpikeBrain 原生 BF16 CE 语义不同，禁用 |

RoPE-only 的两个 2 卡、30-step ACP 重复实验如下。两轮均为相同 seed、batch、GC 和
节点本地 cache；表内依次为 mean / median / p90。

| 实验 | 30 steps | steps 11-30 | steps 21-30 |
| --- | ---: | ---: | ---: |
| v1 native | 26.900 / 31.285 / 40.257s | 22.844 / 25.115 / 36.001s | 19.977 / 18.453 / 31.814s |
| v1 RoPE Liger | 24.303 / 20.566 / 41.461s | 19.390 / 15.154 / 36.702s | 15.561 / 11.260 / 28.911s |
| v2 native | 26.800 / 31.023 / 40.345s | 22.578 / 24.831 / 36.291s | 19.724 / 18.432 / 31.404s |
| v2 RoPE Liger | 24.272 / 20.348 / 39.895s | 19.158 / 15.561 / 36.621s | 15.791 / 12.136 / 27.963s |

两轮中 RoPE-only 的 30-step mean 分别改善约 9.7% 和 9.4%；steps 11-30 mean 均改善
约 15.1%；steps 21-30 mean 分别改善约 22.1% 和 19.9%。收益主要来自减少部分动态
shape 路径的编译/调度开销，缓存命中的单步并非全部更快，显存峰值也基本不变。

当前安全默认因此为：

```text
USE_LIGER_KERNEL=true
LIGER_ROPE=true
LIGER_RMS_NORM=false
LIGER_SWIGLU=false
LIGER_FUSED_LINEAR_CROSS_ENTROPY=false
```

不要直接套用 Qwen 的整套 Liger patch，也不要开启后三个未通过数值检查的算子。

## 5. 为什么 SpikeBrain 仍可能慢于 Qwen-VLAct

本轮已经消除了一个很大的非模型开销，但不能据此认为 SpikeBrain 一定与 Qwen 同速。

已确认的结构差异：

- SpikeBrain-VLAct 总参数约 6.612B，其中约 3.896B trainable。
- SpikeBrain text backbone 为 36 层、hidden size 2560。
- Qwen-VLAct 的 VLM 是 Qwen3-VL-4B；Qwen3-0.6B 在两个方案里都只是 action expert。
- Qwen 路径启用了 Liger 的 RoPE、fused linear cross entropy、RMSNorm 和 SwiGLU；
  SpikeBrain 仅启用经过完整模型严格一致性验证的专用 RoPE。
- SpikeBrain 使用动态稀疏 attention/GDN 自定义 kernel，计算量和动态 shape 编译/调度行为不同，不能直接套用 Qwen 的融合 kernel。

Qwen 长任务日志中常见 `model_time` 约 2.6-4.8s，但它来自 8 卡和不同 batch/GAS 配置。当前 SpikeBrain 探针是 2 卡、VLA 16/GPU、VLM 2/GPU、GAS 1，所以不能把两个数简单除一下作为严格的同条件倍率。

可以确定的是：

- 数据加载不是瓶颈，探针的 `data_time` 通常只有毫秒级。
- AFS cache 查询曾是人为引入的主要瓶颈，现已通过 `/tmp` staging 消除。
- 输入动态 shape 决定了明显的逐 step 波动；相同 step 在不同实验中的快慢分布高度一致。
- 关闭 GC 会 OOM，不能作为正式训练的提速方案。
- 要判断 8 卡正式吞吐，必须等正式 8 卡任务实际开始后，以相同 global batch/样本数口径统计。

## 6. 正式 8 卡任务

显示名：`vlact-spikebrain-lr1e5-bf16-8gpu-100k-v3`

资源名：`pt-24qjy0l5`

截至本文写入时状态：`STARTING / NODE=0`，尚未分配节点，也尚未开始训练。

已核对其启动配置：

- 8 x A100-80GB
- 100000 optimizer steps
- VLA BS/GPU = 16
- VLM BS/GPU = 2
- GAS = 1
- VLM LR = `1e-5`
- action LR = `1e-4`
- weight decay = `0.01`
- BF16
- GC = true（launcher 安全默认）
- pack = false
- camera bucket = false
- W&B offline
- logging frequency = 100
- Liger：仅 SpikeBrain 专用 RoPE；RMSNorm、SwiGLU、fused CE 均关闭

该任务从共享源码启动，因此排到节点后会使用已经验证成功的 `/tmp` 本地 cache staging 方案。

## 7. 验证与检查

已完成：

- 相关 Python 单元测试通过。
- `py_compile` 通过。
- launcher 和探针脚本 `bash -n` 通过。
- 2 卡 GC 本地 cache 探针完成 30 steps，ACP 状态为 SUCCEEDED。
- 三类 gradient norm 在预定日志 step 均存在。
- BF16 训练未触发 non-finite 检查错误。
- 完整模型 RoPE-only forward/loss/backward 零差异验证通过。
- 两轮 2 卡 30-step RoPE-only A/B 均成功，整体 mean 约改善 9.4%-9.7%。

仍待完成：

- 等正式 8 卡任务获得节点。
- 核对 8 卡前 30/100 步的 mean、median、p90 和 samples/s。
- 确认 8 卡显存峰值和长时间稳定性。
- 在相同 batch、相同数据 step、相同卡数口径下，与 Qwen-VLAct 做最终吞吐比较。

## 8. 当前结论

1. 关闭 GC 不可用于当前 6.6B SpikeBrain + ZeRO-2 正式配置，多个缩 batch/缩 bucket 方案均实际 OOM。
2. 持久 kernel cache 是有效的，但训练运行期不能直接使用 AFS cache 目录。
3. “AFS 持久副本 -> `/tmp` 本地运行 -> 结束写回 AFS”已由 30-step ACP 任务完整验证，前 30 步平均时间从旧基线 52.40s 降至 16.15s。
4. grad norm 统计仍完整，但只在真实日志 step 计算，避免每步额外同步。
5. 正式 8 卡任务保留 GC，并采用已验证的本地 cache staging；当前唯一阻塞是 ACP 尚未为它分配 8 卡节点。
6. SpikeBrain 只启用专用 RoPE Liger：数值严格一致且两轮真实训练均提速；RMSNorm、SwiGLU 和 fused CE 因数值不一致保持关闭。
