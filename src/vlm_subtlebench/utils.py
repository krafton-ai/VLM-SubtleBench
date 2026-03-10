"""Configurable base class and config parsing utilities."""

import argparse
import base64
import io
import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Optional, Union
from datetime import datetime

import requests
from omegaconf import DictConfig, OmegaConf


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file or URL to base64 string."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path, timeout=30)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def open_image(image_path: str):
    """Open an image from a file path or URL, returning a PIL Image."""
    from PIL import Image
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    return Image.open(image_path)


def parse_structured(
    fields: Any, cfg: Optional[Union[dict, DictConfig]] = None
) -> Any:
    """Merge structured config defaults with overrides."""
    scfg = OmegaConf.merge(OmegaConf.structured(fields), cfg)
    return scfg

class Configurable:
    """Base class for configurable components."""

    @dataclass
    class Config:
        pass

    cfg: Config

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        self.cfg = parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        raise NotImplementedError
