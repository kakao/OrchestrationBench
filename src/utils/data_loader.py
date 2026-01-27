"""
Data loader for evaluation datasets.

This module handles loading and managing evaluation data from YAML files
based on the configuration patterns specified in the config files.
"""

import glob
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class DataType(Enum):
    """Supported data types for evaluation."""
    YAML = "yaml"
    JSON = "json"
    CSV = "csv"


@dataclass
class EvaluationStep:
    """Individual step in an evaluation scenario."""
    workflow_id: Optional[str]
    step_id: int
    agent: str
    agent_type: str
    data: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationStep':
        """Create EvaluationStep from dictionary."""
        return cls(
            workflow_id=data.get('workflow_id'),
            step_id=data['step_id'],
            agent=data['agent'],
            agent_type=data['agent_type'],
            data=data['data']
        )


@dataclass
class EvaluationScenario:
    """Complete evaluation scenario with metadata and steps."""
    metadata: Dict[str, Any]
    steps: List[EvaluationStep]
    file_path: str
    scenario_id: str
    
    @property
    def total_steps(self) -> int:
        """Get total number of steps in the scenario."""
        return len(self.steps)
    
    @property
    def total_steps(self) -> int:
        """Get total number of steps in the scenario."""
        return len(self.steps)


class DataLoader:
    """Loads and manages evaluation data from various sources."""
    
    def __init__(self, base_path: str = ".", verbose: bool = False):
        """
        Initialize data loader.
        
        Args:
            base_path: Base path for resolving relative paths
            verbose: Enable verbose logging
        """
        self.base_path = Path(base_path)
        self.scenarios: Dict[str, EvaluationScenario] = {}
        self.scenarios_by_type: Dict[str, List[EvaluationScenario]] = {}
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    def load_from_config(self, config: Dict[str, Any]) -> None:
        """
        Load evaluation data based on configuration.
        
        Args:
            config: Configuration dictionary containing eval_data section
        """
        eval_config = config.get('eval_data', {})
        data_type = eval_config.get('type', 'yaml')
        path_pattern = eval_config.get('path', '')
        
        if not path_pattern:
            logger.warning("No path pattern specified in eval_data config")
            return
        
        # Resolve relative path based on base_path (config file location)
        if not Path(path_pattern).is_absolute():
            # Go up one level from config directory to project root
            project_root = self.base_path.parent
            path_pattern = str(project_root / path_pattern)
        
        logger.info(f"Loading {data_type} data from pattern: {path_pattern}")
        
        if data_type == DataType.YAML.value:
            self._load_yaml_files(path_pattern)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _load_yaml_files(self, path_pattern: str) -> None:
        """Load YAML files matching the pattern."""
        yaml_files = glob.glob(path_pattern)
        
        if not yaml_files:
            logger.warning(f"No YAML files found matching pattern: {path_pattern}")
            return
        
        logger.info(f"Found {len(yaml_files)} YAML files to load")
        
        for file_path in sorted(yaml_files):
            try:
                self._load_single_yaml(file_path)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
    
    def _load_single_yaml(self, file_path: str) -> None:
        """Load a single YAML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            logger.warning(f"Empty YAML file: {file_path}")
            return
        
        # Extract metadata and steps
        metadata = data.get('metadata', {})
        steps_data = data.get('steps', [])
        
        if not steps_data:
            logger.warning(f"No steps found in {file_path}")
            return
        
        # Convert steps to EvaluationStep objects
        steps = []
        for step_data in steps_data:
            try:
                step = EvaluationStep.from_dict(step_data)
                steps.append(step)
            except Exception as e:
                logger.error(f"Failed to parse step in {file_path}: {e}")
                continue
        
        # Create scenario
        scenario_id = Path(file_path).stem
        scenario = EvaluationScenario(
            metadata=metadata,
            steps=steps,
            file_path=file_path,
            scenario_id=scenario_id
        )
        
        # Store scenario
        self.scenarios[scenario_id] = scenario
        
        # Group by sheet name or other metadata
        sheet_name = metadata.get('sheet_name', 'default')
        if sheet_name not in self.scenarios_by_type:
            self.scenarios_by_type[sheet_name] = []
        self.scenarios_by_type[sheet_name].append(scenario)
        
        if self.verbose:
            logger.info(f"Loaded scenario {scenario_id}: {scenario.total_steps} steps")
    
    def get_all_scenarios(self) -> List[EvaluationScenario]:
        """Get all loaded scenarios."""
        return list(self.scenarios.values())


def load_evaluation_data(config_path: str, verbose: bool = False) -> DataLoader:
    """
    Convenience function to load evaluation data from config file.
    
    Args:
        config_path: Path to configuration YAML file
        verbose: Enable verbose logging
        
    Returns:
        Configured DataLoader instance
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get base path from config file location
    base_path = Path(config_path).parent
    
    loader = DataLoader(base_path=str(base_path), verbose=verbose)
    loader.load_from_config(config)
    
    return loader