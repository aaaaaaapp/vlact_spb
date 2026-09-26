#!/usr/bin/env bash
# VLAct 预训练数据下载清单（与集群 scripts/setup/download_pretrain_assets.sh 对齐）
# 每项格式: name|repo|repo_type|relative_target|include_globs(逗号分隔,空=全量)
#
# include 说明:
#   - 空 = 下载整个 repo
#   - 多个 glob 用逗号分隔，传给 hf download 的多个 --include

VLACT_DATASET_ENTRIES=(
  # --- Base models（全量）---
  "qwen3_vl_action|StarVLA/Qwen3-VL-4B-Instruct-Action|model|playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action|"
  "qwen3_0_6b|Qwen/Qwen3-0.6B|model|playground/Pretrained_models/Qwen3-0.6B|"

  # --- VLM（部分下载）---
  # 训练只需 zip + json；images/ 由 postprocess unzip 生成
  "llava_coco|StarVLA/LLaVA-OneVision-COCO|dataset|playground/Datasets/LLaVA-OneVision-COCO|sharegpt4v_coco.zip,llava_jsons/**"
  # 训练只需 data/ 下 parquet
  "llava_recap_cc3m|lmms-lab/LLaVA-ReCap-CC3M|dataset|playground/Datasets/LLaVA-ReCap-CC3M|data/**"

  # --- VLA（部分 / 全量）---
  "droid|cadene/droid|dataset|playground/Datasets/DROID|"
  # 小数据集；也可从 RoboInter 派生，见 postprocess prepare_droid_100.py
  "droid_100|lerobot/droid_100|dataset|playground/Datasets/DROID_100|"
  # 只需 household + tabletop 两个 subset（convert 脚本会生成 *_v21）
  "molmoact|allenai/MolmoAct-Dataset|dataset|playground/Datasets/MolmoAct-Dataset|molmoact_dataset_household/train/**,molmoact_dataset_tabletop/train/**"
  # 只需 DROID 标注子集（用于 DROID_100 wrapper + 部分 VLA）
  "robointer_droid|InternRobotics/RoboInter-Data|dataset|playground/Datasets/RoboInter-Data|Annotation_with_action_lerobotv21/lerobot_droid_anno/**"

  # --- RoboCOIN：每个 task 是独立 repo，全量下该 task ---
  "robocoin_cobot_magic_clean_blackboard|RoboCOIN/Cobot_Magic_clean_blackboard|dataset|playground/Datasets/RoboCOIN/Cobot_Magic_clean_blackboard|"
  "robocoin_cobot_magic_fold_the_towel|RoboCOIN/Cobot_Magic_fold_the_towel|dataset|playground/Datasets/RoboCOIN/Cobot_Magic_fold_the_towel|"
  "robocoin_cobot_magic_move_the_bread|RoboCOIN/Cobot_Magic_move_the_bread|dataset|playground/Datasets/RoboCOIN/Cobot_Magic_move_the_bread|"
  "robocoin_cobot_magic_place_the_cup|RoboCOIN/Cobot_Magic_place_the_cup|dataset|playground/Datasets/RoboCOIN/Cobot_Magic_place_the_cup|"
  "robocoin_cobot_magic_stack_the_blocks|RoboCOIN/Cobot_Magic_stack_the_blocks|dataset|playground/Datasets/RoboCOIN/Cobot_Magic_stack_the_blocks|"
  "robocoin_split_aloha_basket_storage_peach|RoboCOIN/Split_aloha_basket_storage_peach|dataset|playground/Datasets/RoboCOIN/Split_aloha_basket_storage_peach|"
  "robocoin_split_aloha_fold_the_towel|RoboCOIN/Split_aloha_fold_the_towel|dataset|playground/Datasets/RoboCOIN/Split_aloha_fold_the_towel|"
  "robocoin_split_aloha_move_the_bread|RoboCOIN/Split_aloha_move_the_bread|dataset|playground/Datasets/RoboCOIN/Split_aloha_move_the_bread|"
  "robocoin_split_aloha_place_the_cup|RoboCOIN/Split_aloha_place_the_cup|dataset|playground/Datasets/RoboCOIN/Split_aloha_place_the_cup|"
  "robocoin_split_aloha_stack_the_blocks|RoboCOIN/Split_aloha_stack_the_blocks|dataset|playground/Datasets/RoboCOIN/Split_aloha_stack_the_blocks|"
)

# InternData-A1 不在此列表：仅 10 个 split_aloha tar.gz，用 03_download_internedata.sh

dataset_entry_by_name() {
  local want="$1"
  local entry name
  for entry in "${VLACT_DATASET_ENTRIES[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "${entry}"
    if [[ "${name}" == "${want}" ]]; then
      echo "${entry}"
      return 0
    fi
  done
  return 1
}

dataset_list_names() {
  local entry name
  for entry in "${VLACT_DATASET_ENTRIES[@]}"; do
    IFS='|' read -r name _ _ _ _ <<< "${entry}"
    echo "${name}"
  done
}

# 分组（方便按需下载）
VLACT_DATASET_GROUPS=(
  "models:qwen3_vl_action,qwen3_0_6b"
  "vlm:llava_coco,llava_recap_cc3m"
  "vla_core:droid,molmoact,robointer_droid"
  "vla_optional:droid_100"
  "robocoin:robocoin_cobot_magic_clean_blackboard,robocoin_cobot_magic_fold_the_towel,robocoin_cobot_magic_move_the_bread,robocoin_cobot_magic_place_the_cup,robocoin_cobot_magic_stack_the_blocks,robocoin_split_aloha_basket_storage_peach,robocoin_split_aloha_fold_the_towel,robocoin_split_aloha_move_the_bread,robocoin_split_aloha_place_the_cup,robocoin_split_aloha_stack_the_blocks"
)

resolve_dataset_group() {
  local group="$1"
  local g spec names
  for g in "${VLACT_DATASET_GROUPS[@]}"; do
    spec="${g#*:}"
    if [[ "${g%%:*}" == "${group}" ]]; then
      names="${spec//,/ }"
      echo "${names}"
      return 0
    fi
  done
  return 1
}
