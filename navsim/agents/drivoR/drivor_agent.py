from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
import re
from .drivor_model import DrivoRModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    ProgressBar,
    LearningRateMonitor,
)
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from .drivor_features import DrivoRTargetBuilder
from .drivor_features import DrivoRFeatureBuilder
import sys
from omegaconf import OmegaConf
import math

# 每个 epoch 的 checkpoint 只保留最近 KEEP_LAST_N 轮 (由 KeepLastNCheckpoints 实现)
KEEP_LAST_N = 10



class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True
    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - train {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - val {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(self, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k,v in metrics.items():
            if "train/" in k:
                train_metrics[k]=v
            elif "val/" in k:
                val_metrics[k]=v
            else:
                other_metrics[k]=v
        print(f"\n###########  Epoch {trainer.current_epoch} ##########")
        for k,v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")


class KeepLastNCheckpoints(Callback):
    """每个 epoch 结束后, 只保留编号最大的 N 个 epoch-{epoch:02d}.ckpt。

    PL 自带的 ModelCheckpoint 只能按指标保留 save_top_k 个 (最好/最差),
    无法表达 "保留最近 N 轮"。此回调配合
    ModelCheckpoint(save_top_k=-1, filename='epoch-{epoch:02d}') 使用:
    先全量写出, 再删除编号超出窗口的旧文件。
    """

    def __init__(self, keep_last_n: int, filename_prefix: str = "epoch-"):
        super().__init__()
        self.keep_last_n = int(keep_last_n)
        self.filename_prefix = filename_prefix
        self._pattern = re.compile(
            rf"{re.escape(filename_prefix)}(?:epoch=)?(\d+).*\.ckpt$"
        )
        # 由 get_training_callbacks 注入, 避免依赖 trainer.checkpoint_callback
        # (PL 只把它指向其中一个 ModelCheckpoint, 未必是 every-epoch 这个)
        self._checkpoint_callback = None

    def _matched_epochs(self, checkpoint_dir: str):
        """返回 [(epoch, path)], 按 epoch 升序。"""
        found = []
        for name in os.listdir(checkpoint_dir):
            m = self._pattern.search(name)
            if m is not None:
                found.append((int(m.group(1)), os.path.join(checkpoint_dir, name)))
        found.sort(key=lambda item: item[0])
        return found

    def on_train_epoch_end(self, trainer, pl_module):
        # 只在主进程执行, 避免 DDP 各 rank 重复删除
        if trainer.is_global_zero is False:
            return
        ckpt_cb = self._checkpoint_callback or getattr(
            trainer, "checkpoint_callback", None
        )
        checkpoint_dir = getattr(ckpt_cb, "dirpath", None)
        if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
            return
        matched = self._matched_epochs(checkpoint_dir)
        for epoch, path in matched[: max(0, len(matched) - self.keep_last_n)]:
            try:
                os.remove(path)
                print(f"[KeepLastNCheckpoints] removed {os.path.basename(path)}", flush=True)
            except FileNotFoundError:
                pass


class DrivoRAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            warm_start_path: str = "",
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
    ):
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        # 仅在训练时用于热启动权重 (strict=False 加载)。
        # 与 checkpoint_path 解耦: checkpoint_path 同时承担 "非空即 eval 模式"
        # 的哨兵语义 (见下方 self.loss 的初始化条件), warm_start_path 不参与
        # 任何模式判断, 因此训练时传它不会破坏 self.loss / self.ray 的构建。
        self._warm_start_path = warm_start_path
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus


        cache_data=False

        if not cache_data:
            self._drivor_model = DrivoRModel(config)

        if not cache_data and self._checkpoint_path == "": # only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray=True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(threads_per_node=8)
                self.worker_map=worker_map


            from .score_module.compute_navsim_score import get_scores

            metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
            try:
                # add synthetic metric_cache
                metric_cache_synthetic_0 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-0"))
                metric_cache_synthetic_1 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-1"))
                metric_cache_synthetic_2 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-2"))
                metric_cache_synthetic_3 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-3"))
                metric_cache_synthetic_4 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-4"))

                self.train_metric_cache_paths_synthetic = metric_cache_synthetic_0.metric_cache_paths
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_0.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_1.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_2.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_3.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_4.metric_cache_paths)

                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
            except:
                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic = None

            self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths

            self.get_scores = get_scores

            self.loss = loss
            


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        # eval 模式用 checkpoint_path; 训练热启动用 warm_start_path。
        # 两者互斥使用, eval 优先级更高以保持原有行为不变。
        load_path = (
            self._checkpoint_path
            if self._checkpoint_path not in (None, "")
            else self._warm_start_path
        )

        if load_path not in (None, ""):
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(load_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(load_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            # self.load_state_dict({k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()})
            result = self.load_state_dict(
                {
                    k.replace("agent._drivor_model", "_drivor_model"): v
                    for k, v in state_dict.items()
                },
                strict=False,
            )
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)

    # def get_sensor_config(self) :
    #     """Inherited, see superclass."""
    #     # return SensorConfig(
    #     #     cam_f0=[3],
    #     #     cam_l0=[3],
    #     #     cam_l1=[],
    #     #     cam_l2=[],
    #     #     cam_r0=[3],
    #     #     cam_r1=[],
    #     #     cam_r2=[],
    #     #     cam_b0=[3],
    #     #     lidar_pc=[],
    #     # )
    #     return SensorConfig(
    #         cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
    #         cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
    #         cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
    #         cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
    #         cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
    #         cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
    #         cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
    #         cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
    #         lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
    #     )

    # def _select_world_model_action(
    #     self,
    #     pred: Dict[str, torch.Tensor],
    #     targets: Dict[str, torch.Tensor],
    # ) -> torch.Tensor:
    #     proposals = pred["proposals"]        # [B, M, T, 3]
    #     gt = targets["trajectory"]           # [B, T, 3]

    #     # Use the same basic distance convention as DrivoR trajectory training.
    #     match_error = torch.linalg.norm(
    #         proposals - gt[:, None],
    #         dim=-1,
    #         ord=1,
    #     ).mean(-1)                           # [B, M]

    #     # Discrete matching itself is not differentiable.
    #     best_idx = match_error.detach().argmin(dim=1)

    #     batch_idx = torch.arange(
    #         proposals.shape[0],
    #         device=proposals.device,
    #     )

    #     # IMPORTANT: no detach here; world loss can update trajectory decoder.
    #     return proposals[batch_idx, best_idx]
    
    
    def _select_world_model_action(
        self,
        pred: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Soft-match the predicted proposals to the GT trajectory.

        Every proposal is weighted by how close it is to the GT trajectory
        (soft-argmin over the L1 trajectory distance). The weights themselves
        are detached, but the proposals are NOT, so the world-model loss can
        backpropagate into ALL proposals of the trajectory decoder instead of
        only the single hard-argmin winner.
        """

        proposals = pred["proposals"]
        # [B, M, T, 3]

        gt = targets["trajectory"].to(proposals.dtype)
        # [B, T, 3]
        # 混合精度 (16-mixed) 下 proposals 为 fp16, 而 targets 常为 fp32/fp64。
        # 若不显式对齐, 相减会把结果提升为 fp64, softmax 出的 weights 也是 fp64,
        # 随后 einsum(weights, proposals) 因 dtype 不一致报
        # "expected scalar type Half but found Double"。

        # Same style of L1 trajectory distance used by DrivoR training.
        # 全程保持 proposals 的 dtype, 避免中间量被提升。
        proposals_detached = proposals.detach()
        match_error = torch.linalg.norm(
            proposals_detached - gt[:, None],
            ord=1,
            dim=-1,
        ).mean(dim=-1)
        # [B, M]

        temperature = float(
            self._config.world_model.get("action_temperature", 1.0)
        )

        # Lower match_error -> higher weight. Detached: the matching is a
        # soft target assignment, not a differentiable objective.
        weights = torch.softmax(
            -match_error / temperature,
            dim=1,
        ).to(proposals.dtype)
        # [B, M]

        # Weighted average over all proposals; gradient reaches every proposal.
        positive_trajectory = torch.einsum(
            "bm,bmtd->btd",
            weights,
            proposals,
        )

        # IMPORTANT: do not detach positive_trajectory.
        return positive_trajectory



    def _sensor_indices_with_future(self, key):
        """相机索引列表; world model 启用时额外追加未来帧索引作为监督目标。"""
        indices = list(OmegaConf.to_object(self._config[key]))

        if (
            self._config.world_model.enabled
            and len(indices) > 0
        ):
            current_idx = max(indices)
            future_idx = (
                current_idx
                + self._config.world_model.future_offset_frames
            )
            if future_idx not in indices:
                indices.append(future_idx)

        return sorted(indices)

    def get_sensor_config(self):
        return SensorConfig(
            cam_f0=self._sensor_indices_with_future("cam_f0"),
            cam_l0=self._sensor_indices_with_future("cam_l0"),
            cam_l1=self._sensor_indices_with_future("cam_l1"),
            cam_l2=self._sensor_indices_with_future("cam_l2"),
            cam_r0=self._sensor_indices_with_future("cam_r0"),
            cam_r1=self._sensor_indices_with_future("cam_r1"),
            cam_r2=self._sensor_indices_with_future("cam_r2"),
            cam_b0=self._sensor_indices_with_future("cam_b0"),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )
    
    def get_target_builders(self) :
        return [DrivoRTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [DrivoRFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._drivor_model(features)

    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
            metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
        else:
            metric_cache_paths = self.test_metric_cache_paths
            metric_cache_paths_synthetic = self.test_metric_cache_paths_synthetic

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        
        data_points = [
            {
                "token": metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token],
                "poses": poses,
                "test": test
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    # def compute_loss(
    #         self,
    #         features: Dict[str, torch.Tensor],
    #         targets: Dict[str, torch.Tensor],
    #         pred: Dict[str, torch.Tensor],
    # ) -> Dict:
    #     return self.loss(targets, pred, self._config, self.compute_score)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        current_epoch: int = None,
    ) -> Dict:
        """DrivoR loss + 可选 (curriculum) LAW-style world-model 辅助损失。

        current_epoch 由 AgentLightningModule.training_step 透传 (PL 内置属性,
        DDP 下各 rank 一致)。validation / inference 不会传它, 因此该分支
        天然只可能在训练时被激活。
        """

        # ============================================================
        # World-model curriculum guard
        #
        # enabled          : 整个实验是否包含 world model
        # start_epoch      : 从哪个 epoch 起真正启用该 training branch
        #
        # 两者语义分离: enabled 不在训练中动态改写, 因此不影响任何
        # 与初始化相关的逻辑 (world model 始终在 __init__ 中创建,
        # 参数始终在 optimizer 内)。
        # ============================================================
        world_model_active = bool(
            self.training
            and self._config.world_model.enabled
            and current_epoch is not None
            and current_epoch >= self._config.world_model.start_epoch
        )

        # 供 DrivoRLoss 记录日志 (epoch 0~9: active=0 / weight=0.0)
        pred["world_model_active"] = float(world_model_active)
        pred["world_loss_weight_effective"] = (
            float(self.loss.world_loss_weight)
            if world_model_active
            else 0.0
        )

        # 每个 epoch 只在第一个 batch 打印一次, 便于直接确认 curriculum 切换
        if current_epoch is not None and current_epoch != getattr(
            self, "_wm_last_logged_epoch", None
        ):
            self._wm_last_logged_epoch = current_epoch
            print(
                f"[WorldModel] epoch={current_epoch} "
                f"active={world_model_active} "
                f"weight="
                f"{pred['world_loss_weight_effective']}",
                flush=True,
            )

        # ============================================================
        # LAW-style register world model branch
        # TRAINING ONLY + 仅在 current_epoch >= start_epoch 时执行
        #
        # epoch < start_epoch 时该块整体跳过:
        #   - future image encoder 不执行
        #   - world model forward 不执行
        #   - world loss 不计算 (DrivoRLoss 回落到 0)
        # 因此 world model 参数在该阶段不会产生任何梯度。
        # ============================================================
        if world_model_active:

            current_registers = pred[
                "current_registers"
            ]
            # Expected:
            # [B, N_registers, D]
            # e.g. [B, 64, 256]

            # --------------------------------------------------------
            # 1. Build validity mask
            # --------------------------------------------------------
            world_valid = targets[
                "world_valid"
            ].to(
                device=current_registers.device,
                dtype=torch.bool,
                non_blocking=True,
            )

            # --------------------------------------------------------
            # 2. Only run world-model branch if at least one
            #    sample in this batch has a complete future frame.
            # --------------------------------------------------------
            if bool(world_valid.any().item()):

                valid_idx = torch.nonzero(
                    world_valid,
                    as_tuple=False,
                ).squeeze(1)

                # ----------------------------------------------------
                # 3. Select predicted positive trajectory
                # ----------------------------------------------------
                action_trajectory = (
                    self._select_world_model_action(
                        pred,
                        targets,
                    )
                )
                # [B, T, 3]

                valid_current_registers = (
                    current_registers.index_select(
                        0,
                        valid_idx,
                    )
                )

                valid_action_trajectory = (
                    action_trajectory.index_select(
                        0,
                        valid_idx,
                    )
                )

                # ----------------------------------------------------
                # 4. LAW world-model prediction
                #
                # current registers + predicted trajectory
                #        -> predicted future registers
                # ----------------------------------------------------
                world_pred = (
                    self._drivor_model
                    .predict_future_registers(
                        valid_current_registers,
                        valid_action_trajectory,
                    )
                )

                # ----------------------------------------------------
                # 5. Select valid future RGB samples ONLY.
                #
                # This fixes pitfall #2 as well:
                # invalid samples never run through future DINO.
                # ----------------------------------------------------
                future_image = targets[
                    "future_image"
                ].to(
                    device=current_registers.device,
                    non_blocking=True,
                )

                valid_future_image = (
                    future_image.index_select(
                        0,
                        valid_idx,
                    )
                )

                # ----------------------------------------------------
                # 6. Future target encoder
                #
                # detach_target=True:
                #   - forward is still computed
                #   - no backward graph
                #   - substantially lower activation memory
                #
                # apply_grid_mask=False:
                #   keep target representation deterministic/stable
                # ----------------------------------------------------
                if (
                    self._config.world_model.detach_target
                ):
                    with torch.no_grad():
                        world_target = (
                            self._drivor_model
                            .encode_camera_registers(
                                valid_future_image,
                                apply_grid_mask=False,
                            )
                        )

                else:
                    world_target = (
                        self._drivor_model
                        .encode_camera_registers(
                            valid_future_image,
                            apply_grid_mask=False,
                        )
                    )

                # ----------------------------------------------------
                # 7. Safety checks
                # ----------------------------------------------------
                if (
                    world_pred.shape
                    != world_target.shape
                ):
                    raise RuntimeError(
                        "World-model shape mismatch: "
                        f"pred={tuple(world_pred.shape)}, "
                        f"target={tuple(world_target.shape)}"
                    )

                if not torch.isfinite(
                    world_pred
                ).all():
                    raise RuntimeError(
                        "Non-finite values found in world_pred."
                    )

                if not torch.isfinite(
                    world_target
                ).all():
                    raise RuntimeError(
                        "Non-finite values found in world_target."
                    )

                # ----------------------------------------------------
                # 8. Pass to DrivoRLoss
                # ----------------------------------------------------
                pred["world_pred"] = world_pred
                pred["world_target"] = world_target

                # Useful for logging/debugging.
                pred["world_valid_rate"] = (
                    world_valid.float().mean()
                )

            else:
                # Entire batch has no valid future target.
                # Do NOT create world_pred/world_target.
                # DrivoRLoss will automatically set world_loss = 0.
                pred["world_valid_rate"] = (
                    world_valid.float().mean()
                )

        # ============================================================
        # Original DrivoR loss
        # ============================================================
        return self.loss(
            targets,
            pred,
            self._config,
            self.compute_score,
        )

    def get_optimizers(self):

        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.Adam(self._drivor_model.parameters(), lr=lr)
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.AdamW(self._drivor_model.parameters(), lr=lr)
        else:
            raise NotImplementedError

        if self.scheduler_args is not None:

            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)

            # classic cosine
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=T_max, 
            #     eta_min=0.0, last_epoch=-1
            # )

            # Ramp + cosine
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_cosine, 
                eta_min=0.0, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )           

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):

        checkpoint_cb_best = ModelCheckpoint(save_top_k=1,
                                        monitor='val/score_epoch',
                                        filename='best-{epoch}-{step}',
                                        mode="max"
                                        )
        
        checkpoint_cb = ModelCheckpoint(save_last=True)

        # 每个 epoch 保存一个独立 checkpoint (filename=epoch-{epoch:02d});
        # save_top_k=-1 先全量写出, 再由 KeepLastNCheckpoints 只保留最近
        # KEEP_LAST_N 轮 (PL 的 save_top_k 只能按指标保留最好/最差, 无法表达
        # "最近 N 轮")。monitor/mode 仅用于让 state_key 与上面两个
        # ModelCheckpoint 互不相同 (PL 要求同类型 stateful callback 的
        # state_key 唯一); save_top_k=-1 下不参与筛选。
        checkpoint_cb_every_epoch = ModelCheckpoint(
            filename='epoch-{epoch:02d}',
            save_top_k=-1,
            every_n_epochs=1,
            monitor='val/score_epoch',
            mode="min",
        )

        keep_last_n_cb = KeepLastNCheckpoints(keep_last_n=KEEP_LAST_N)
        keep_last_n_cb._checkpoint_callback = checkpoint_cb_every_epoch

        lr_monitor = LearningRateMonitor(logging_interval="step",
                                            log_momentum=False,
                                            log_weight_decay=False)

        if self.progress_bar:
            return [checkpoint_cb_best, checkpoint_cb, checkpoint_cb_every_epoch,
                    keep_last_n_cb, lr_monitor]
        else:
            progress_bar = LitProgressBar()
            return [checkpoint_cb_best, checkpoint_cb, checkpoint_cb_every_epoch,
                    keep_last_n_cb, progress_bar, lr_monitor]
