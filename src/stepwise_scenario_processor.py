"""
Scenario Processor

This module combines data_loader and simple_history_generator to process
evaluation scenarios from YAML files and generate JSON history outputs.
"""

import asyncio
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.progress import Progress, TaskID

from src.utils.data_loader import DataLoader, EvaluationScenario
from src.utils.config_loader import load_config
from src.step_history_generator import SimpleHistoryGenerator
from loguru import logger


console = Console()

class ScenarioProcessor:
    """Processes evaluation scenarios and generates history outputs"""
    
    def __init__(self, 
                 config_file: Path, 
                 agent_cards_dir: Path,
                 output_dir: Path = Path("data/results/step_wise_evaluation"),
                 model: str = "custom_local",
                 batch_size: int = 5,
                 max_retries: int = 1):
        """
        Initialize the scenario processor.
        
        Args:
            config_file: Path to the YAML configuration file
            agent_cards_dir: Path to directory containing agent card JSON files
            output_dir: Directory to save output JSON files
            model: Model type to use
            batch_size: Maximum number of concurrent agent executions per batch
            max_retries: Maximum number of retry attempts for failed agent executions
        """
        self.config_file = config_file
        self.agent_cards_dir = agent_cards_dir
        self.output_dir = output_dir
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        config = load_config(self.config_file)
        try:
            self.model_name = config.get("llms").get(self.model).get("model")
        except:
            logger.error(f"No model config found as name of {self.model}")
            raise
        # Ensure output directory exists
        # Replace path separators to avoid nested directories
        safe_model_name = self.model_name.replace("/", "_").replace("\\", "_")
        self.output_dir = Path(self.output_dir) / safe_model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

        logger.debug(f"🎯 Scenario Processor initialized", style="bold blue")
        logger.debug(f"📁 Output directory: {self.output_dir}", style="dim")
        
    async def initialize(self):
        """Initialize components"""
        # Initialize SimpleHistoryGenerator
        self.history_generator = SimpleHistoryGenerator(
            config_file=self.config_file,
            agent_cards_dir=self.agent_cards_dir,
            model_type=self.model,
            batch_size=self.batch_size,
            max_retries=self.max_retries
        )
        await self.history_generator.initialize()
        logger.debug("✅ History generator initialized", style="green")

    async def close(self):
        """Close all resources and model sessions."""
        if self.history_generator:
            await self.history_generator.close()
            logger.debug("✅ History generator closed", style="green")

    def load_scenarios(self, data_path: str = "data/evaluation/*.yaml") -> List[EvaluationScenario]:
        """
        Load evaluation scenarios from YAML files.
        
        Args:
            data_path: Glob pattern for YAML files
            
        Returns:
            List of loaded scenarios
        """
        # Create temporary config for data loader
        temp_config = {
            "eval_data": {
                "type": "yaml", 
                "path": data_path
            }
        }
        
        # Initialize data loader with config file directory as base path
        base_path = self.config_file.parent
        data_loader = DataLoader(base_path=str(base_path), verbose=True)
        data_loader.load_from_config(temp_config)
        
        scenarios = data_loader.get_all_scenarios()
        logger.debug(f"📋 Loaded {len(scenarios)} scenarios", style="cyan")
        
        return scenarios
    
    def merge_usage_stats(self, usage_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple usage statistics into a single consolidated usage dict.
        
        Args:
            usage_list: List of usage dictionaries from multiple runs
            
        Returns:
            Merged usage statistics
        """
        if not usage_list:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
        
        merged = {
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        
        for usage in usage_list:
            # Extract the simplified usage stats we need
            if "total" in usage:
                total_stats = usage["total"]
                merged["total_calls"] += total_stats.get("total_calls", 0)
                merged["total_input_tokens"] += total_stats.get("total_input_tokens", 0)
                merged["total_output_tokens"] += total_stats.get("total_output_tokens", 0)
                merged["total_tokens"] += total_stats.get("total_tokens", 0)
                merged["total_cost"] += total_stats.get("total_cost", 0.0)
            else:
                # If the usage dict is already in simplified format
                merged["total_calls"] += usage.get("total_calls", 0)
                merged["total_input_tokens"] += usage.get("total_input_tokens", 0)
                merged["total_output_tokens"] += usage.get("total_output_tokens", 0)
                merged["total_tokens"] += usage.get("total_tokens", 0)
                merged["total_cost"] += usage.get("total_cost", 0.0)
        
        return merged
    
    async def process_scenario(self, scenario: EvaluationScenario, num_iterations: int = 1) -> Dict[str, Any]:
        """
        Process a single scenario and generate history for multiple iterations.
        
        Args:
            scenario: EvaluationScenario to process
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary containing all runs and merged usage statistics
        """
        logger.debug(f"🔄 Processing scenario: {scenario.scenario_id} ({num_iterations} iterations)", style="yellow")
        
        all_histories = {}
        all_usage = []
        
        try:
            for i in range(num_iterations):
                logger.debug(f"   🔄 Run #{i+1}/{num_iterations}", style="dim")
                
                # Process evaluation steps using SimpleHistoryGenerator
                history, usage = await self.history_generator.process_evaluation_steps(scenario.steps)
                
                # Store this run's history
                all_histories[f"run #{i+1}"] = history
                all_usage.append(usage)
                
                logger.debug(f"   ✅ Run #{i+1} completed", style="dim green")
            
            # Merge all usage statistics
            merged_usage = self.merge_usage_stats(all_usage)
            
            logger.debug(f"✅ Scenario {scenario.scenario_id} processed successfully ({num_iterations} runs)", style="green")
            logger.debug(f"   - Total merged calls: {merged_usage['total_calls']}", style="dim")
            logger.debug(f"   - Total merged tokens: {merged_usage['total_tokens']}", style="dim")
            
            return all_histories, merged_usage
            
        except Exception as e:
            logger.debug(f"❌ Error processing scenario {scenario.scenario_id}: {e}", style="red")
            raise

    def save_history(self, scenario_id: str, histories: Dict[str, Any], usage: Dict[str, Any]) -> Path:
        """
        Save histories and usage to JSON file.
        
        Args:
            scenario_id: Scenario identifier
            histories: Dictionary containing all run histories
            usage: Merged usage statistics
            
        Returns:
            Path to saved file
        """
        output_file = self.output_dir / f"{scenario_id}_out.json"
        
        # Add metadata to the output
        output_data = {
            "scenario_id": scenario_id,
            "generated_at": str(asyncio.get_event_loop().time()),
            "history": histories,
            "usage": usage
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"💾 Saved: {output_file}", style="green")
        return output_file
    
    async def process_all_scenarios(self, 
                                   data_path: str = "data/evaluation/*.yaml",
                                   max_scenarios: Optional[int] = None,
                                   num_iterations: int = 1) -> List[Path]:
        """
        Process all scenarios and save results.
        
        Args:
            data_path: Glob pattern for YAML files
            max_scenarios: Maximum number of scenarios to process (None for all)
            num_iterations: Number of iterations to run per scenario
            
        Returns:
            List of paths to saved JSON files
        """
        logger.debug(f"🚀 Starting scenario processing... ({num_iterations} iterations per scenario)", style="bold blue")
        
        # Load scenarios
        scenarios = self.load_scenarios(data_path)
        
        if max_scenarios:
            scenarios = scenarios[:max_scenarios]
            logger.debug(f"🎯 Processing first {max_scenarios} scenarios", style="yellow")
        
        if not scenarios:
            logger.debug("⚠️ No scenarios found!", style="yellow")
            return []
        
        saved_files = []
        
        # Semaphore for batch control
        semaphore = asyncio.Semaphore(self.batch_size)

        async def _bounded_process(scenario, progress_bar, task_started, task_completed):
            """Internal wrapper to process scenario with semaphore and progress update"""
            async with semaphore:
                try:
                    # Process scenario with multiple iterations
                    progress_bar.update(task_started, advance=1)

                    histories, usage = await self.process_scenario(scenario, num_iterations)

                    # Save to JSON
                    output_file = self.save_history(scenario.scenario_id, histories, usage)
                    return output_file
                except Exception as e:
                    logger.debug(f"💥 Failed to process {scenario.scenario_id}: {e}", style="red")
                    return None
                finally:
                    progress_bar.update(task_completed, advance=1)

        # Process with progress bar (single Progress instance with two tasks)
        with Progress() as progress:
            task_started = progress.add_task(
                "[cyan] STARTED Processing scenarios...",
                total=len(scenarios)
            )
            task_completed = progress.add_task(
                "[green] COMPLETED Processing scenarios...",
                total=len(scenarios)
            )

            # Create tasks for all scenarios
            tasks = [_bounded_process(scenario, progress, task_started, task_completed) for scenario in scenarios]

            # Execute concurrently
            results = await asyncio.gather(*tasks)

            # Collect successful results
            saved_files = [res for res in results if res is not None]
        
        # Summary
        logger.debug("\n" + "="*60, style="bold cyan")
        logger.debug("📊 PROCESSING SUMMARY", style="bold cyan")
        logger.debug("="*60, style="bold cyan")
        logger.debug(f"Total scenarios: {len(scenarios)}", style="cyan")
        logger.debug(f"Iterations per scenario: {num_iterations}", style="cyan")
        logger.debug(f"Successfully processed: {len(saved_files)}", style="green")
        logger.debug(f"Failed: {len(scenarios) - len(saved_files)}", style="red")
        logger.debug(f"Output directory: {self.output_dir}", style="dim")
        logger.debug("="*60, style="bold cyan")
        
        return saved_files
    
    async def process_single_scenario_by_id(self, scenario_id: str, data_path: str = "data/evaluation/*.yaml", num_iterations: int = 1) -> Optional[Path]:
        """
        Process a single scenario by its ID.
        
        Args:
            scenario_id: ID of the scenario to process
            data_path: Glob pattern for YAML files
            num_iterations: Number of iterations to run
            
        Returns:
            Path to saved JSON file, or None if scenario not found
        """
        scenarios = self.load_scenarios(data_path)
        
        target_scenario = None
        for scenario in scenarios:
            if scenario.scenario_id == scenario_id:
                target_scenario = scenario
                break
        
        if not target_scenario:
            logger.debug(f"❌ Scenario '{scenario_id}' not found", style="red")
            return None
        
        try:
            histories, usage = await self.process_scenario(target_scenario, num_iterations)
            output_file = self.save_history(scenario_id, histories, usage)
            return output_file
        except Exception as e:
            logger.debug(f"💥 Failed to process {scenario_id}: {e}", style="red")
            return None


async def process_scenarios(config_file: str = "config/base_config/multiagent_config.yaml",
                           agent_cards_dir: str = "config/multiagent_cards",
                           data_path: str = "data/evaluation/*.yaml",
                           output_dir: str = "data/results/step_wise_evaluation",
                           max_scenarios: Optional[int] = None,
                           num_iterations: int = 1,
                           batch_size: int = 5,
                           max_retries: int = 1) -> List[Path]:
    """
    Convenience function to process scenarios.
    
    Args:
        config_file: Path to configuration file
        agent_cards_dir: Path to agent cards directory
        data_path: Glob pattern for evaluation YAML files
        output_dir: Output directory for JSON files
        max_scenarios: Maximum number of scenarios to process
        num_iterations: Number of iterations to run per scenario
        
    Returns:
        List of paths to saved JSON files
    """
    processor = ScenarioProcessor(
        config_file=Path(config_file),
        agent_cards_dir=Path(agent_cards_dir),
        output_dir=Path(output_dir),
        batch_size=batch_size,
        max_retries=max_retries
    )
    
    await processor.initialize()
    return await processor.process_all_scenarios(data_path, max_scenarios, num_iterations)


if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Process evaluation scenarios and generate history outputs.")
        parser.add_argument("--config", default="config/base_config/multiagent_config.yaml", 
                          help="Path to configuration file")
        parser.add_argument("--agent-cards", default="data/KO/multiagent_cards",
                          help="Path to agent cards directory")
        parser.add_argument("--data-path", default="data/KO/scenario_data/*.yaml",
                          help="Glob pattern for evaluation YAML files")
        parser.add_argument("--output-dir", default="data/results/step_wise_evaluation",
                          help="Output directory for JSON files")
        parser.add_argument("--max-scenarios", type=int,
                          help="Maximum number of scenarios to process")
        parser.add_argument("--scenario-id", 
                          help="Process only a specific scenario by ID")
        parser.add_argument("--model", default="custom_local",
                          help="Model to use for processing")
        parser.add_argument("--num-iter", type=int, default=1,
                          help="Number of iterations for processing")
        parser.add_argument("--batch-size", type=int, default=5,
                          help="Maximum number of concurrent agent executions per batch")
        parser.add_argument("--max-retries", type=int, default=10,
                          help="Maximum number of retry attempts for failed agent executions")
        parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                          default="INFO", help="Set logging level")

        args = parser.parse_args()

        # Configure logger based on log level
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level=args.log_level)

        logger.debug(f"Parsed arguments: {args}", style="blue")
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=args.batch_size,))
        try:
            processor = ScenarioProcessor(
                config_file=Path(args.config),
                agent_cards_dir=Path(args.agent_cards),
                output_dir=Path(args.output_dir),
                model=args.model,
                batch_size=args.batch_size,
                max_retries=args.max_retries
            )
            
            await processor.initialize()
            
            if args.scenario_id:
                # Process single scenario
                result = await processor.process_single_scenario_by_id(
                    args.scenario_id, args.data_path, args.num_iter
                )
                if result:
                    logger.debug(f"✅ Successfully processed scenario: {result}", style="bold green")
                else:
                    logger.debug("❌ Failed to process scenario", style="bold red")
            else:
                # Process all scenarios
                results = await processor.process_all_scenarios(
                    args.data_path, args.max_scenarios, args.num_iter
                )
                logger.debug(f"🎉 Processing completed! Generated {len(results)} files.", style="bold green")

        except Exception as e:
            logger.debug(f"💥 Processing failed: {e}", style="bold red")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up resources
            await processor.close()
    
    asyncio.run(main())
