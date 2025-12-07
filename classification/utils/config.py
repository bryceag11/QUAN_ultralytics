#!/usr/bin/env python
import os
import yaml
import datetime
import copy
import json
from pathlib import Path
import shutil


class Config:
    """Configuration handler for training experiments"""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file.
                         If None, uses default configuration.
        """
        # Load default configuration
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "default.yaml"
        )
        
        with open(default_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Override with specific configuration if provided
        if config_path is not None:
            with open(config_path, 'r') as f:
                specific_config = yaml.safe_load(f)
            
            # Deep merge specific config into default config
            self._deep_update(self.config, specific_config)
            
        # Set up experiment directory
        self._setup_experiment_dir()
        
    def _deep_update(self, d, u):
        """
        Recursively update dict d with values from dict u
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = copy.deepcopy(v)
    
    def _setup_experiment_dir(self):
        """
        Create experiment directory structure
        """
        # Get experiment name or create one based on date and number
        if self.config['experiment']['name'] == 'default':
            # Get current date
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            
            # Find latest experiment number for today
            exp_num = 1
            experiments_dir = Path("experiments")
            if experiments_dir.exists():
                existing_exps = [d for d in experiments_dir.iterdir() 
                                if d.is_dir() and d.name.startswith(date_str)]
                exp_num = len(existing_exps) + 1
            
            # Create new experiment name
            exp_name = f"{date_str}_{exp_num:02d}_{self.config['model']['type']}_{self.config['model']['name']}"
            self.config['experiment']['name'] = exp_name
        
        # Create experiment directory
        self.exp_dir = Path("experiments") / self.config['experiment']['name']
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.logs_dir = self.exp_dir / "logs"
        self.visualizations_dir = self.exp_dir / "visualizations"
        
        # Create directories
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        
        # Save configuration
        self.save_config()
    
    def save_config(self, path=None):
        """
        Save current configuration to YAML file
        
        Args:
            path: Path to save the file. If None, saves to experiment dir.
        """
        if path is None:
            path = self.exp_dir / "config.yaml"
        
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, *keys, default=None):
        """
        Get configuration value for a nested key
        
        Example: 
            config.get('training', 'learning_rate')
            
        Args:
            *keys: Keys to navigate the configuration dictionary
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value or default
        """
        result = self.config
        try:
            for key in keys:
                result = result[key]
            return result
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key):
        """
        Allow direct access to top-level configuration keys
        
        Example:
            config['training']
        """
        return self.config[key]
    
    def __str__(self):
        """String representation of the configuration"""
        return json.dumps(self.config, indent=2)
    
    def update_key(self, value, *keys):
        """
        Update a specific configuration key
        
        Args:
            value: New value to set
            *keys: Keys to navigate the configuration dictionary
        """
        if not keys:
            return
        
        # Navigate to the right nesting level
        config_level = self.config
        for key in keys[:-1]:
            if key not in config_level:
                config_level[key] = {}
            config_level = config_level[key]
        
        # Set the value
        config_level[keys[-1]] = value
        
        # Save updated configuration
        self.save_config()


def load_config_for_inference(config_path):
    """
    Load configuration for inference only
    
    Args:
        config_path: Path to the saved configuration YAML
        
    Returns:
        Config object with the loaded configuration
    """
    cfg = Config()
    
    with open(config_path, 'r') as f:
        loaded_config = yaml.safe_load(f)
    
    # Replace the default config with the loaded one
    cfg.config = loaded_config
    
    return cfg