import toml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from clifford.utils import ensure_directory
from clifford.exceptions import ConfigurationError


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    shuffle: bool = True
    verbose: bool = True


@dataclass
class DatabaseConfig:
    path: str = "database/clifford.db"
    backup_enabled: bool = True
    backup_interval: int = 3600


@dataclass
class ModelConfig:
    save_path: str = "models"
    checkpoint_interval: int = 10
    auto_save: bool = True


@dataclass
class GUIConfig:
    theme: str = "dark"
    window_width: int = 1200
    window_height: int = 800
    auto_refresh: bool = True
    refresh_interval: int = 5


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)


class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "clifford.toml"
        self.config_path = Path(config_path)
        self.config = Config()

    def load(self) -> Config:
        if not self.config_path.exists():
            self.save_default()
            return self.config
        
        try:
            with open(self.config_path, 'r') as f:
                data = toml.load(f)
            
            if "training" in data:
                self.config.training = TrainingConfig(**data["training"])
            if "database" in data:
                self.config.database = DatabaseConfig(**data["database"])
            if "model" in data:
                self.config.model = ModelConfig(**data["model"])
            if "gui" in data:
                self.config.gui = GUIConfig(**data["gui"])
            
            return self.config
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}")

    def save(self) -> None:
        ensure_directory(self.config_path.parent)
        
        data = {
            "training": {
                "epochs": self.config.training.epochs,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "validation_split": self.config.training.validation_split,
                "early_stopping_patience": self.config.training.early_stopping_patience,
                "shuffle": self.config.training.shuffle,
                "verbose": self.config.training.verbose
            },
            "database": {
                "path": self.config.database.path,
                "backup_enabled": self.config.database.backup_enabled,
                "backup_interval": self.config.database.backup_interval
            },
            "model": {
                "save_path": self.config.model.save_path,
                "checkpoint_interval": self.config.model.checkpoint_interval,
                "auto_save": self.config.model.auto_save
            },
            "gui": {
                "theme": self.config.gui.theme,
                "window_width": self.config.gui.window_width,
                "window_height": self.config.gui.window_height,
                "auto_refresh": self.config.gui.auto_refresh,
                "refresh_interval": self.config.gui.refresh_interval
            }
        }
        
        try:
            with open(self.config_path, 'w') as f:
                toml.dump(data, f)
        except Exception as e:
            raise ConfigurationError(f"Failed to save config: {e}")

    def save_default(self) -> None:
        self.config = Config()
        self.save()

    def get(self, key: str) -> Any:
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                raise ConfigurationError(f"Config key not found: {key}")
        
        return value

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        obj = self.config
        
        for k in keys[:-1]:
            if hasattr(obj, k):
                obj = getattr(obj, k)
            else:
                raise ConfigurationError(f"Config key not found: {key}")
        
        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], value)
        else:
            raise ConfigurationError(f"Config key not found: {key}")

    def update(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            self.set(key, value)
