from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DecoderHead(nn.Module):
    def __init__(
        self,
        feature_channels: list[int] | None = None,
        embedding_dim: int = 256,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = [112, 224, 448, 896]

        self.linear_c4 = MLP(input_dim=feature_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=feature_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=feature_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=feature_channels[0], embed_dim=embedding_dim)

        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(embedding_dim, output_dim, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        batch_size = c4.shape[0]

        c4_out = self.linear_c4(c4.flatten(2).transpose(1, 2)).transpose(1, 2).view(
            batch_size,
            -1,
            c4.shape[2],
            c4.shape[3],
        )
        c4_out = F.interpolate(c4_out, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c3_out = self.linear_c3(c3.flatten(2).transpose(1, 2)).transpose(1, 2).view(
            batch_size,
            -1,
            c3.shape[2],
            c3.shape[3],
        )
        c3_out = F.interpolate(c3_out, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c2_out = self.linear_c2(c2.flatten(2).transpose(1, 2)).transpose(1, 2).view(
            batch_size,
            -1,
            c2.shape[2],
            c2.shape[3],
        )
        c2_out = F.interpolate(c2_out, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c1_out = self.linear_c1(c1.flatten(2).transpose(1, 2)).transpose(1, 2).view(
            batch_size,
            -1,
            c1.shape[2],
            c1.shape[3],
        )

        fused = self.linear_fuse(torch.cat([c4_out, c3_out, c2_out, c1_out], dim=1))
        fused = self.bn(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        return self.classifier(fused)


class RefinerBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + residual
        return self.relu(out)


class CNNRefinerModule(nn.Module):
    def __init__(self, in_channels: int = 7, hidden_channels: int = 64, out_channels: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res1 = RefinerBlock(hidden_channels, dilation=1)
        self.res2 = RefinerBlock(hidden_channels, dilation=2)
        self.res3 = RefinerBlock(hidden_channels, dilation=4)
        self.res4 = RefinerBlock(hidden_channels, dilation=8)
        self.final = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        nn.init.normal_(self.final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.final.bias, 0)

    def forward(self, image: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, coarse_pred], dim=1)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.final(x) * 10.0


class GreenFormer(nn.Module):
    def __init__(
        self,
        encoder_name: str = "hiera_base_plus_224.mae_in1k_ft_in1k",
        in_channels: int = 4,
        img_size: int = 2048,
        use_refiner: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for CorridorKey. Install requirements.txt in the ComfyUI Python environment."
            ) from exc

        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, img_size=img_size)
        if in_channels != 3:
            self._patch_input_layer(in_channels)

        try:
            feature_channels = self.encoder.feature_info.channels()
        except (AttributeError, TypeError):
            feature_channels = [112, 224, 448, 896]

        self.alpha_decoder = DecoderHead(feature_channels, 256, output_dim=1)
        self.fg_decoder = DecoderHead(feature_channels, 256, output_dim=3)

        self.use_refiner = use_refiner
        self.refiner = CNNRefinerModule(in_channels=7, hidden_channels=64, out_channels=4) if use_refiner else None

    def _patch_input_layer(self, in_channels: int) -> None:
        try:
            patch_embed = self.encoder.model.patch_embed.proj
        except AttributeError:
            patch_embed = self.encoder.patch_embed.proj

        weight = patch_embed.weight.data
        bias = patch_embed.bias.data if patch_embed.bias is not None else None
        out_channels, _, kernel_h, kernel_w = weight.shape

        new_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_h, kernel_w),
            stride=patch_embed.stride,
            padding=patch_embed.padding,
            bias=(bias is not None),
        )
        new_conv.weight.data[:, :3, :, :] = weight
        new_conv.weight.data[:, 3:, :, :] = 0.0
        if bias is not None:
            new_conv.bias.data = bias

        try:
            self.encoder.model.patch_embed.proj = new_conv
        except AttributeError:
            self.encoder.patch_embed.proj = new_conv

    def load_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        new_state_dict: dict[str, torch.Tensor] = {}
        model_state = self.state_dict()

        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                key = key[10:]

            if "pos_embed" in key and key in model_state and value.shape != model_state[key].shape:
                src_tokens = value.shape[1]
                dst_tokens = model_state[key].shape[1]
                channels = value.shape[2]
                src_grid = int(math.sqrt(src_tokens))
                dst_grid = int(math.sqrt(dst_tokens))
                reshaped = value.permute(0, 2, 1).view(1, channels, src_grid, src_grid)
                resized = F.interpolate(reshaped, size=(dst_grid, dst_grid), mode="bicubic", align_corners=False)
                value = resized.flatten(2).transpose(1, 2)

            new_state_dict[key] = value

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        return list(missing), list(unexpected)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        features = self.encoder(x)

        alpha_logits = self.alpha_decoder(features)
        fg_logits = self.fg_decoder(features)

        alpha_logits_up = F.interpolate(alpha_logits, size=input_size, mode="bilinear", align_corners=False)
        fg_logits_up = F.interpolate(fg_logits, size=input_size, mode="bilinear", align_corners=False)

        alpha_coarse = torch.sigmoid(alpha_logits_up)
        fg_coarse = torch.sigmoid(fg_logits_up)
        rgb = x[:, :3, :, :]
        coarse_pred = torch.cat([alpha_coarse, fg_coarse], dim=1)

        if self.use_refiner and self.refiner is not None:
            delta_logits = self.refiner(rgb, coarse_pred)
        else:
            delta_logits = torch.zeros_like(coarse_pred)

        alpha_final = torch.sigmoid(alpha_logits_up + delta_logits[:, 0:1])
        fg_final = torch.sigmoid(fg_logits_up + delta_logits[:, 1:4])
        return {"alpha": alpha_final, "fg": fg_final}
