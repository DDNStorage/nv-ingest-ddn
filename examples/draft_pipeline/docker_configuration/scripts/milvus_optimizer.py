#!/usr/bin/env python3
"""
Milvus Bulk Load Optimizer
Main script to generate optimized Docker Compose configurations
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader

# Add lib directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.resource_detector import ResourceDetector
from modules.bulk_load_optimizer import BulkLoadOptimizer
from modules.storage_config import StorageConfig


class MilvusOptimizer:
    """Main optimizer class that orchestrates the configuration generation"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.template_dir = self.base_dir / 'templates'
        self.config_dir = self.base_dir / 'configs'
        
        # Set up Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.base_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def parse_embeddings(self, embeddings_str: str) -> Optional[int]:
        """Parse embedding count from string (e.g., '20M', '1.5B')"""
        if not embeddings_str:
            return None
        
        multipliers = {
            'K': 1_000,
            'M': 1_000_000,
            'B': 1_000_000_000
        }
        
        # Remove spaces and convert to uppercase
        embeddings_str = embeddings_str.strip().upper()
        
        # Check for multiplier suffix
        for suffix, multiplier in multipliers.items():
            if embeddings_str.endswith(suffix):
                try:
                    num = float(embeddings_str[:-1])
                    return int(num * multiplier)
                except ValueError:
                    print(f"Invalid embedding count: {embeddings_str}")
                    return None
        
        # Try to parse as plain number
        try:
            return int(embeddings_str)
        except ValueError:
            print(f"Invalid embedding count: {embeddings_str}")
            return None
    
    def generate_configuration(self, target_embeddings: Optional[int] = None, 
                             output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate optimized configuration"""
        
        # Detect system resources
        print("Detecting system resources...")
        resources = ResourceDetector.detect_all()
        ResourceDetector.print_summary(resources)
        
        # Get storage configuration
        print("\nLoading storage configuration...")
        storage_config = StorageConfig.get_storage_config()
        print(f"Storage type: {storage_config['type']}")
        
        # Create optimizer and calculate configuration
        optimizer = BulkLoadOptimizer(resources)
        
        if target_embeddings:
            print(f"\nOptimizing for {target_embeddings:,} embeddings...")
            config = optimizer.calculate_target_mode(target_embeddings)
        else:
            print("\nUsing auto mode (maximum performance)...")
            config = optimizer.calculate_auto_mode()
        
        # Validate configuration
        if not optimizer.validate_configuration(config):
            print("Warning: Configuration may exceed available resources!")
        
        # Get summary
        summary = optimizer.get_resource_summary(config, target_embeddings)
        
        # Prepare template context
        context = {
            'resources': resources,
            'config': config,
            'storage': storage_config,
            'summary': summary
        }
        
        # Generate files
        if output_dir is None:
            output_dir = self.base_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Milvus configuration
        print("\nGenerating Milvus configuration...")
        milvus_template = self.jinja_env.get_template('configs/milvus-bulk-optimized.yaml.j2')
        milvus_content = milvus_template.render(**context)
        milvus_path = output_dir / 'configs' / 'milvus.yaml'
        milvus_path.parent.mkdir(exist_ok=True)
        with open(milvus_path, 'w') as f:
            f.write(milvus_content)
        print(f"Created: {milvus_path}")
        
        # Generate Docker Compose file
        print("Generating Docker Compose file...")
        compose_template = self.jinja_env.get_template('templates/docker-compose-bulk.yml.j2')
        compose_content = compose_template.render(**context)
        compose_path = output_dir / 'docker-compose.yml'
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        print(f"Created: {compose_path}")
        
        # Generate summary JSON
        summary_path = output_dir / 'optimization-summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'resources': resources,
                'config': config,
                'storage': storage_config,
                'summary': summary
            }, f, indent=2)
        print(f"Created: {summary_path}")
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print configuration summary"""
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Mode: {summary['mode']}")
        print(f"Total nodes: {summary['total_nodes']}")
        print(f"  - Index nodes: {summary['node_distribution']['index_nodes']}")
        print(f"  - Data nodes: {summary['node_distribution']['data_nodes']}")
        print(f"  - Query nodes: {summary['node_distribution']['query_nodes']}")
        print(f"\nMemory allocation:")
        print(f"  - Total: {summary['memory']['total_allocated_gb']} GB")
        print(f"  - Per node: {summary['memory']['per_node_mb']} MB")
        print(f"  - System utilization: {summary['memory']['utilization_percent']}%")
        print(f"\nParallelism settings:")
        for key, value in summary['parallelism'].items():
            print(f"  - {key}: {value}")
        if 'gpu' in summary:
            print(f"\nGPU configuration:")
            print(f"  - GPU count: {summary['gpu']['count']}")
            print(f"  - Initial memory: {summary['gpu']['init_mem_size_mb']} MB")
            print(f"  - Max memory: {summary['gpu']['max_mem_size_mb']} MB")
        print("\nSegment configuration:")
        print(f"  - Max size: {summary['segment_config']['max_size_mb']} MB")
        print(f"  - Seal proportion: {summary['segment_config']['seal_proportion']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Milvus Bulk Load Optimizer - Generate optimized configurations'
    )
    parser.add_argument(
        '--target-embeddings',
        type=str,
        help='Target number of embeddings (e.g., 20M, 1.5B). If not specified, uses auto mode.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for generated files (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without generating files'
    )
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = MilvusOptimizer()
    
    # Parse target embeddings if provided
    target_embeddings = None
    if args.target_embeddings:
        target_embeddings = optimizer.parse_embeddings(args.target_embeddings)
        if target_embeddings is None:
            sys.exit(1)
    
    # Generate configuration
    if args.dry_run:
        # Just show the configuration
        resources = ResourceDetector.detect_all()
        bulk_optimizer = BulkLoadOptimizer(resources)
        
        if target_embeddings:
            config = bulk_optimizer.calculate_target_mode(target_embeddings)
        else:
            config = bulk_optimizer.calculate_auto_mode()
        
        summary = bulk_optimizer.get_resource_summary(config, target_embeddings)
        optimizer.print_summary(summary)
    else:
        # Generate files
        summary = optimizer.generate_configuration(target_embeddings, args.output_dir)
        optimizer.print_summary(summary)
        print(f"\nConfiguration files generated successfully!")
        print("To start Milvus, run: docker-compose up -d")


if __name__ == "__main__":
    main()