#!/usr/bin/env python3
"""
Test runner script for the Match Engine project.
Supports running all tests, specific test files, with verbose output,
and optional coverage reporting.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def find_python():
    """Find the Python executable."""
    if sys.executable:
        return sys.executable
    for cmd in ['python3', 'python']:
        try:
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    raise RuntimeError("Python not found. Please install Python 3.")


def check_coverage_installed(python_cmd):
    """Check if coverage is installed, install if not."""
    try:
        subprocess.run(
            [python_cmd, '-m', 'coverage', '--version'],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Coverage not installed. Installing coverage...")
        try:
            subprocess.run(
                [python_cmd, '-m', 'pip', 'install', 'coverage', '--quiet'],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install coverage")
            return False


def run_tests(python_cmd, test_file=None, verbose=False, heavy_tests=False, coverage=False):
    """Run the tests."""
    # Set environment variable for heavy tests
    env = os.environ.copy()
    if heavy_tests:
        env['RUN_HEAVY_DATASET_TESTS'] = '1'
        print("⚠ Heavy dataset tests enabled (this may take longer)")
        print()

    # Build test command
    if test_file:
        # Remove .py extension if provided
        test_name = test_file.replace('.py', '')
        test_path = f"tests.{test_name}"
        print(f"Running specific test: {test_name}")
        print()
        
        cmd = [python_cmd, '-m', 'unittest', test_path]
        if verbose:
            cmd.append('-v')
    else:
        print("Running all tests...")
        print()
        
        cmd = [python_cmd, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py']
        if verbose:
            cmd.append('-v')

    # Run with coverage if requested
    if coverage:
        if not check_coverage_installed(python_cmd):
            print("✗ Cannot run with coverage. Running tests without coverage...")
            coverage = False

    if coverage:
        print("Running tests with coverage...")
        print()
        # Coverage needs to wrap the unittest command
        coverage_cmd = [
            python_cmd, '-m', 'coverage', 'run',
            '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py'
        ]
        if verbose:
            coverage_cmd.append('-v')
        
        result = subprocess.run(coverage_cmd, env=env)
        
        if result.returncode == 0:
            print()
            print("Coverage Report:")
            print()
            subprocess.run([python_cmd, '-m', 'coverage', 'report'])
            print()
            print("✓ Coverage report generated. Open htmlcov/index.html in a browser for detailed report.")
            subprocess.run([python_cmd, '-m', 'coverage', 'html'], capture_output=True)
        
        return result.returncode
    else:
        result = subprocess.run(cmd, env=env)
        return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run tests for the Match Engine project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all tests
  %(prog)s -v                       # Run all tests with verbose output
  %(prog)s -t test_matcher          # Run specific test file
  %(prog)s --heavy                  # Run all tests including heavy dataset tests
  %(prog)s --coverage               # Run tests with coverage report
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Run tests with verbose output'
    )
    
    parser.add_argument(
        '-t', '--test',
        metavar='FILE',
        help='Run a specific test file (e.g., test_matcher)'
    )
    
    parser.add_argument(
        '--heavy',
        action='store_true',
        help='Run heavy dataset tests (sets RUN_HEAVY_DATASET_TESTS=1)'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run tests with coverage report'
    )
    
    args = parser.parse_args()
    
    print("========================================")
    print("Match Engine Test Runner")
    print("========================================")
    print()
    
    try:
        python_cmd = find_python()
        python_version = subprocess.run(
            [python_cmd, '--version'],
            capture_output=True,
            text=True
        ).stdout.strip()
        print(f"✓ Using {python_version}")
        print()
    except RuntimeError as e:
        print(f"✗ {e}")
        sys.exit(1)
    
    exit_code = run_tests(
        python_cmd,
        test_file=args.test,
        verbose=args.verbose,
        heavy_tests=args.heavy,
        coverage=args.coverage
    )
    
    if exit_code == 0:
        print()
        print("========================================")
        print("All tests passed!")
        print("========================================")
    else:
        print()
        print("========================================")
        print("Some tests failed!")
        print("========================================")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

