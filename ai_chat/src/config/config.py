import yaml
import os
from pathlib import Path
from typing import Dict, Any, List

class Config:
    def __init__(self):
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the config file
        config_path = os.path.join(current_dir, 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Store model families configuration
        self.model_families = config.get('model_families', {})
        
        # Store prompts configuration
        self.prompts = config.get('prompts', {})
    
    def get_model_family(self, family_name: str) -> dict:
        """Get configuration for a specific model family"""
        return self.model_families.get(family_name, {})
    
    def get_model_config(self, family_name: str, model_name: str) -> dict:
        """Get configuration for a specific model within a family"""
        family = self.get_model_family(family_name)
        return family.get('models', {}).get(model_name, {})
    
    def get_prompt_config(self, prompt_name: str) -> dict:
        """Get configuration for a specific prompt"""
        return self.prompts.get(prompt_name, {})

    @property
    def functions(self) -> Dict[str, Any]:
        return self._config.get('functions', {})
    
    @property
    def image_processing(self) -> Dict[str, Any]:
        """Get image processing configuration"""
        return self._config.get('image_processing', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get any configuration value by key"""
        return self._config.get(key, default)

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get specific model configuration"""
        return self.models.get(model_key, {})

    def get_prompt_info(self, prompt_key: str) -> Dict[str, Any]:
        """Get specific prompt configuration"""
        return self.prompts.get(prompt_key, {})