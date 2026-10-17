import torch
import torch.nn as nn


class LAWRegisterWorldModel(nn.Module):
    """
    LAW-style latent world model adapted to DrivoR camera registers.

    registers:  [B, N_register, D]
    trajectory: [B, T, 3]  (x, y, heading)
    output:      [B, N_register, D]
    """

    def __init__(
        self,
        d_model: int,
        num_poses: int,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        use_heading: bool = False,
    ):
        super().__init__()
        self.use_heading = use_heading

        action_dim_per_pose = 3 if use_heading else 2
        action_dim = num_poses * action_dim_per_pose

        # LAW-style action-aware latent:
        # a_t^i = MLP([z_t^i, trajectory])
        self.action_fusion = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # LAW describes 2 Transformer blocks with self-attention + FFN.
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.world_model = nn.TransformerEncoder(
            block,
            num_layers=num_layers,
        )

    def forward(
        self,
        registers: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = registers.shape

        if not self.use_heading:
            trajectory = trajectory[..., :2]

        # [B, T, 2/3] -> [B, action_dim]
        action = trajectory.flatten(start_dim=1)

        # Same ego action conditions every register.
        action = action[:, None, :].expand(-1, N, -1)

        action_aware_registers = self.action_fusion(
            torch.cat([registers, action], dim=-1)
        )

        return self.world_model(action_aware_registers)