from __future__ import annotations

from monai.networks.nets import UNet


def build_monai_unet3d(config: dict) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        channels=tuple(int(v) for v in config["channels"]),
        strides=tuple(int(v) for v in config["strides"]),
        num_res_units=int(config["num_res_units"]),
    )
