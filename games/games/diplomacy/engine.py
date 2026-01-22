# -*- coding: utf-8 -*-
"""Diplomacy engine config."""

from __future__ import annotations
import yaml
import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DiplomacyConfig:
    """AvalonBasicConfig-like config for Diplomacy."""
    power_names: List[str]
    map_name: str = "standard"
    max_phases: int = 20
    negotiation_rounds: int = 3
    seed: int = 42
    language: str = "en"
    human_power: Optional[str] = None 
    roles: Optional[dict] = None 

    @classmethod
    def default(cls) -> "DiplomacyConfig":
        """
        Prioritize reading defaults from yaml (if available), otherwise use hardcoded defaults.
        Support specifying yaml path via environment variable DIPLOMACY_CONFIG_YAML.
        """
        # 1. Default parameters
        base = dict(
            power_names=["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"],
            map_name="standard",
            max_phases=20,
            negotiation_rounds=3,
            seed=42,
            language="en",
            human_power=None,
        )

        # 2. Try to read from yaml
        yaml_path = os.environ.get("DIPLOMACY_CONFIG_YAML", "games/games/diplomacy/configs/default_config.yaml")
        roles = None
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yml = yaml.safe_load(f) or {}
                game_cfg = yml.get('game', {}) if isinstance(yml, dict) else {}
                roles = yml.get('roles', None)
                for k in base:
                    if k in game_cfg and game_cfg[k] is not None:
                        base[k] = game_cfg[k]
            except Exception as e:
                print(f"[DiplomacyConfig] Failed to load yaml: {e}")

        return cls(**base, roles=roles)
