# starVLA/model/modules/action_model/FlowRAM_ActionHeader.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from typing import Optional, Dict, Any

class FlowRAMActionHead(nn.Module):
    """
    简化版 FlowRAM 动作头（无 CUDA 依赖）
    核心思想：
      1. 保留 Conditional Flow Matching 框架（4步推理）
      2. 用 2D 区域注意力近似 DRS（Dynamic Radius Schedule）
      3. 用线性融合替代 Mamba（避免 CUDA 依赖）
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 通用参数（与 GR00T 兼容）
        # self.action_dim = getattr(config, "action_dim", 14)  # RoboTwin 双臂 = 14
        self.action_dim = getattr(config, "action_dim", 7)
        self.hidden_size = getattr(config, "hidden_size", 1024)
        # self.action_horizon = getattr(config, "action_horizon", 16)
        self.action_horizon = getattr(config, "action_horizon", 8)
        
        # === 新增：VLM 特征投影层（2560 → 1024）===
        # 从配置获取 VLM 输出维度（Qwen3-VL = 2560, Qwen2.5 = 2048）
        self.vl_hidden_dim = getattr(config, "vl_hidden_dim", 2560)  # 默认 2560 (Qwen3)
        self.vl_proj = nn.Linear(self.vl_hidden_dim, self.hidden_size)
        # =========================================

        # 1. 区域感知注意力（2D 近似 DRS）
        self.region_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # 2. 使用 Mamba 做融合（不再用线性融合替代）
        self.fusion_in = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fusion_norm = nn.LayerNorm(self.hidden_size)
        self.fusion_mamba = Mamba(
            d_model=self.hidden_size,
            d_state=getattr(config, "mamba_d_state", 16),
            d_conv=getattr(config, "mamba_d_conv", 4),
            expand=getattr(config, "mamba_expand", 2),
        )
        self.fusion_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        
        # 3. 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # 4. 动作投影层（action_dim -> hidden_size）
        self.action_proj = nn.Linear(self.action_dim, self.hidden_size)
        # self.action_proj = nn.Linear(7, self.hidden_size)
        
        # 5. 速度场解码器（CFM 核心）
        self.velocity_decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.action_dim)
        )
        
        # 6. 位置编码（用于序列位置）
        self.pos_embed = nn.Embedding(1024, self.hidden_size)
    
    # def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, 
    #             state: Optional[torch.Tensor] = None,
    #             encoder_attention_mask: Optional[torch.Tensor] = None):
    #     """
    #     训练阶段：Conditional Flow Matching 损失
        
    #     Args:
    #         vl_embs: [B, L, H] - VLM 视觉-语言特征
    #         actions: [B, T, action_dim] - 目标动作序列
    #     """
    #     B, T, _ = actions.shape
        
    #     # === CFM 标准流程 ===
    #     # 1. 生成噪声和时间步
    #     noise = torch.randn_like(actions)
    #     t = torch.rand(B, 1, 1, device=actions.device)  # [B, 1, 1]
    #     noisy_actions = (1 - t) * noise + t * actions   # x_t = (1-t)*x0 + t*x1
    #     target_velocity = actions - noise               # v = x1 - x0
        
    #     # 2. 时间嵌入
    #     time_emb = self.time_embed(t * 1000)  # [B, 1, H]
        
    #     # 3. 动作特征编码
    #     action_features = self.action_proj(noisy_actions)  # [B, T, H]
    #     action_features = action_features + time_emb       # 加入时间信息
        
    #     # === 关键修改：投影 VLM 特征到 hidden_size ===
    #     vl_embs_proj = self.vl_proj(vl_embs)  # [B, L, 2560] → [B, L, 1024]
    #     # ===========================================

    #     # 4. 区域感知注意力（2D 近似 DRS：聚焦任务相关区域）
    #     attended_features, _ = self.region_attention(
    #         query=action_features,
    #         key=vl_embs_proj,  # ✅ 使用投影后的特征
    #         value=vl_embs_proj,
    #         attn_mask=encoder_attention_mask
    #     )  # [B, T, H]
        
    #     # 5. Mamba 融合
    #     fused = torch.cat([attended_features, action_features], dim=-1)  # [B, T, 2H]
    #     fused = self.fusion_in(fused)                                     # [B, T, H]
    #     fused = fused + self.fusion_mamba(self.fusion_norm(fused))        # 残差
    #     fused = self.fusion_out(fused)
     
    #     # 6. 预测速度场
    #     pred_velocity = self.velocity_decoder(fused)  # [B, T, action_dim]
        
    #     # 7. Flow Matching 损失
    #     loss = ((pred_velocity - target_velocity) ** 2).mean()
    #     return loss
    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        model_dtype = self.time_embed[0].weight.dtype
        vl_embs = vl_embs.to(dtype=model_dtype)
        actions = actions.to(dtype=model_dtype)

        B, T, _ = actions.shape
        noise = torch.randn_like(actions)
        t = torch.rand(B, 1, 1, device=actions.device, dtype=model_dtype)

        noisy_actions = (1 - t) * noise + t * actions
        target_velocity = actions - noise

        time_emb = self.time_embed(t * 1000)
        action_features = self.action_proj(noisy_actions) + time_emb
        vl_embs_proj = self.vl_proj(vl_embs)

        attended_features, _ = self.region_attention(
            query=action_features,
            key=vl_embs_proj,
            value=vl_embs_proj,
            attn_mask=encoder_attention_mask,
        )

        fused = torch.cat([attended_features, action_features], dim=-1)
        fused = self.fusion_in(fused)
        fused = fused + self.fusion_mamba(self.fusion_norm(fused))
        fused = self.fusion_out(fused)

        pred_velocity = self.velocity_decoder(fused)
        loss = ((pred_velocity.float() - target_velocity.float()) ** 2).mean()
        return loss
    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        推理阶段：4步直接预测
        """
        B = vl_embs.shape[0]
        T = self.action_horizon

        # 统一到模型参数 dtype（关键修复）
        model_dtype = self.time_embed[0].weight.dtype
        device = vl_embs.device
        vl_embs = vl_embs.to(dtype=model_dtype)

        actions = torch.randn(B, T, self.action_dim, device=device, dtype=model_dtype)

        num_steps = getattr(self.config, "num_inference_timesteps", 4)
        dt = torch.tensor(1.0 / num_steps, device=device, dtype=model_dtype)

        vl_embs_proj = self.vl_proj(vl_embs)

        for step in range(num_steps):
            t = torch.full((B, 1, 1), step / num_steps, device=device, dtype=model_dtype)  # 关键：dtype一致

            time_emb = self.time_embed(t * 1000)
            action_features = self.action_proj(actions) + time_emb

            attended, _ = self.region_attention(
                query=action_features,
                key=vl_embs_proj,
                value=vl_embs_proj,
            )

            fused = torch.cat([attended, action_features], dim=-1)
            fused = self.fusion_in(fused)
            fused = fused + self.fusion_mamba(self.fusion_norm(fused))
            fused = self.fusion_out(fused)

            velocity = self.velocity_decoder(fused)
            actions = actions + dt * velocity

        return actions
    # @torch.no_grad()
    # def predict_action(self, vl_embs: torch.Tensor, state: Optional[torch.Tensor] = None):
    #     """
    #     推理阶段：4步直接预测（FlowRAM 核心优势）
        
    #     Args:
    #         vl_embs: [B, L, H] - VLM 特征
    #     Returns:
    #         pred_actions: [B, T, action_dim] - 预测动作
    #     """
    #     B = vl_embs.shape[0]
    #     T = self.action_horizon
        
    #     # 1. 从噪声分布初始化
    #     actions = torch.randn(B, T, self.action_dim, device=vl_embs.device)
        
    #     # 2. 仅需 4 步即可收敛（FlowRAM 优势）
    #     num_steps = getattr(self.config, "num_inference_timesteps", 4)
    #     dt = 1.0 / num_steps
        
    #     # === 关键修改：投影 VLM 特征 ===
    #     vl_embs_proj = self.vl_proj(vl_embs)
    #     # ==============================

    #     for step in range(num_steps):
    #         # 当前时间步 (归一化到 [0,1])
    #         t = torch.full((B, 1, 1), step / num_steps, device=vl_embs.device)
            
    #         # 与训练阶段相同处理
    #         time_emb = self.time_embed(t * 1000)
    #         action_features = self.action_proj(actions) + time_emb
            
    #         # 区域感知 + Mamba 融合
    #         attended, _ = self.region_attention(
    #             query=action_features,
    #             key=vl_embs_proj,
    #             value=vl_embs_proj
    #         )
    #         fused = torch.cat([attended, action_features], dim=-1)
    #         fused = self.fusion_in(fused)
    #         fused = fused + self.fusion_mamba(self.fusion_norm(fused))
    #         fused = self.fusion_out(fused)

    #         # 预测速度场并积分（欧拉法）
    #         velocity = self.velocity_decoder(fused)
    #         actions = actions + dt * velocity  # x_{t+dt} = x_t + v * dt
        
    #     return actions

def get_action_model(config):
    """工厂函数：与 GR00T 接口兼容"""
    return FlowRAMActionHead(config)