"""Configurable base class and config parsing utilities."""

import argparse
import base64
import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Optional, Union
from datetime import datetime

from omegaconf import DictConfig, OmegaConf


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
