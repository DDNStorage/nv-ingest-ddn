#!/usr/bin/env python3
"""
Main Test Orchestrator for Milvus Docker Services
Tests both Infinia and GCS configurations in CPU mode
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class MilvusServiceTester:
    """Main orchestrator for testing Milvus services"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.scripts_dir = base_dir / "scripts"
        self.tests_dir = base_dir / "tests"
        self.results = {}
        
    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{text.center(70)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
    
    def print_section(self, text: str):
        """Print a section header"""
        print(f"\n{Colors.BLUE}{'─'*50}{Colors.ENDC}")
        print(f"{Colors.BLUE}{text}{Colors.ENDC}")
        print(f"{Colors.BLUE}{'─'*50}{Colors.ENDC}")
    
    def print_success(self, text: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")
    
    def print_error(self, text: str):
        """Print error message"""
        print(f"{Colors.RED}✗ {text}{Colors.ENDC}")
    
    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")
    
    def run_command(self, cmd: List[str], description: str, timeout: int = 300) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        print(f"\n{Colors.BLUE}Running: {description}{Colors.ENDC}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.base_dir
            )
            
            if result.returncode == 0:
                self.print_success(f"{description} completed successfully")
                return True, result.stdout
            else:
                self.print_error(f"{description} failed")
                print(f"Error output:\n{result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            self.print_error(f"{description} timed out after {timeout} seconds")
            return False, "Command timed out"
        except Exception as e:
            self.print_error(f"{description} failed with exception: {e}")
            return False, str(e)
    
    def stop_services(self, env: str) -> bool:
        """Stop Milvus services for the given environment"""
        self.print_section(f"Stopping services for {env}")
        
        cmd = ["./scripts/run_milvus.sh", "--env", env, "--stop"]
        success, _ = self.run_command(cmd, f"Stop {env} services", timeout=60)
        
        # Give services time to fully stop
        if success:
            print("Waiting for services to stop completely...")
            time.sleep(10)
        
        return success
    
    def clean_environment(self):
        """Clean up any running containers before tests"""
        self.print_section("Cleaning environment")
        
        # Check for any running Milvus containers
        cmd = ["docker", "ps", "-q", "--filter", "name=milvus-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout.strip():
            self.print_warning("Found running Milvus containers, stopping them...")
            
            # Stop all Milvus containers
            stop_cmd = ["docker", "stop"] + result.stdout.strip().split('\n')
            subprocess.run(stop_cmd, capture_output=True)
            
            # Remove stopped containers
            rm_cmd = ["docker", "rm"] + result.stdout.strip().split('\n')
            subprocess.run(rm_cmd, capture_output=True)
            
            self.print_success("Cleaned up existing containers")
        else:
            self.print_success("No existing Milvus containers found")
    
    def start_services(self, env: str) -> bool:
        """Start Milvus services for the given environment in CPU mode"""
        self.print_section(f"Starting services for {env} in CPU mode")
        
        cmd = ["./scripts/run_milvus.sh", "--env", env, "--cpu"]
        success, output = self.run_command(cmd, f"Start {env} services", timeout=180)
        
        if success:
            # Wait for services to be ready
            print("\nWaiting for services to initialize...")
            for i in range(30, 0, -5):
                print(f"  Waiting {i} seconds...", end='\r')
                time.sleep(5)
            print("  Services should be ready now!    ")
        
        return success
    
    def check_service_health(self) -> bool:
        """Run the service health check script"""
        self.print_section("Checking service health")
        
        cmd = ["./tests/check_service_health.sh"]
        success, output = self.run_command(cmd, "Service health check", timeout=60)
        
        if success:
            print("\nHealth check output:")
            print(output)
        
        return success
    
    def test_milvus_connection(self, env: str) -> bool:
        """Test Milvus connection and operations"""
        self.print_section(f"Testing Milvus connection for {env}")
        
        cmd = ["python3", "./tests/test_milvus_connection.py", "--env", env]
        success, output = self.run_command(cmd, "Milvus connection test", timeout=120)
        
        if success:
            print("\nConnection test output:")
            print(output)
        
        return success
    
    def test_environment(self, env: str) -> Dict[str, bool]:
        """Test a complete environment (start, check health, test connection, stop)"""
        self.print_header(f"Testing {env.upper()} Configuration")
        
        results = {
            "start": False,
            "health": False,
            "connection": False,
            "stop": False
        }
        
        # Start services
        results["start"] = self.start_services(env)
        if not results["start"]:
            self.print_error(f"Failed to start {env} services, skipping remaining tests")
            self.stop_services(env)  # Try to stop anyway
            return results
        
        # Check health
        results["health"] = self.check_service_health()
        
        # Test connection (even if health check has warnings)
        results["connection"] = self.test_milvus_connection(env)
        
        # Stop services
        results["stop"] = self.stop_services(env)
        
        return results
    
    def run_all_tests(self, environments: List[str]):
        """Run tests for all specified environments"""
        self.print_header("Milvus Docker Services Test Suite")
        
        # Initial cleanup
        self.clean_environment()
        
        # Test each environment
        for env in environments:
            self.results[env] = self.test_environment(env)
            
            # Brief pause between environments
            if env != environments[-1]:
                print("\nPausing before next environment test...")
                time.sleep(10)
        
        # Print final summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("Test Summary")
        
        all_passed = True
        
        for env, results in self.results.items():
            print(f"\n{Colors.BOLD}{env.upper()} Configuration:{Colors.ENDC}")
            
            for test, passed in results.items():
                status = f"{Colors.GREEN}✓ PASSED{Colors.ENDC}" if passed else f"{Colors.RED}✗ FAILED{Colors.ENDC}"
                print(f"  {test.capitalize():<15} {status}")
                
                if not passed:
                    all_passed = False
        
        print("\n" + "="*70)
        if all_passed:
            self.print_success("All tests passed successfully!")
        else:
            self.print_error("Some tests failed!")
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Test Milvus Docker services with different storage backends'
    )
    parser.add_argument(
        '--env',
        type=str,
        choices=['infinia', 'gcs', 'both'],
        default='both',
        help='Which environment(s) to test (default: both)'
    )
    parser.add_argument(
        '--skip-cleanup',
        action='store_true',
        help='Skip initial cleanup of existing containers'
    )
    
    args = parser.parse_args()
    
    # Determine base directory
    base_dir = Path(__file__).parent.parent
    
    # Create tester instance
    tester = MilvusServiceTester(base_dir)
    
    # Determine which environments to test
    if args.env == 'both':
        environments = ['infinia', 'gcs']
    else:
        environments = [args.env]
    
    # Run tests
    try:
        tester.run_all_tests(environments)
        
        # Exit with appropriate code
        all_passed = all(
            all(results.values()) 
            for results in tester.results.values()
        )
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.ENDC}")
        print("Attempting to stop any running services...")
        
        for env in environments:
            tester.stop_services(env)
        
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()