"""
Command-line interface for Offarm in PY.

This module provides the CLI entry point for running simulations from the terminal.

Usage:
    # When installed via pip:
    $ offarm run path/to/INPUT_manifest.json
    $ offarm run path/to/INPUT_manifest.json --env env_withoutVar.json --plot
    
    # When running as module:
    $ python -m offarm run path/to/INPUT_manifest.json
    
    # From local folder (usr0):
    $ cd examples/usr0
    $ python -m offarm run INPUT_manifest.json
    
    # Or if offarm is installed:
    $ cd examples/usr0
    $ offarm run INPUT_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="offarm",
        description="Offarmpy - Offshore Farm Laboratory CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default settings
  offarm run INPUT_manifest.json
  
  # Run with custom environment and plotting
  offarm run INPUT_manifest.json --env env_withoutVar.json --plot
  
  # Run with specific case ID and description
  offarm run INPUT_manifest.json --case-id 1 --info "Test case"
  
  # Save results to custom directory
  offarm run INPUT_manifest.json --output-dir my_results
  
  # Quick run without saving (for testing)
  offarm run INPUT_manifest.json --no-save --no-progress

For more information, visit: https://github.com/Hugo-build/offarm-py
""",
    )
    
    parser.add_argument(
        "--version", "-V",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a simulation from manifest file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "manifest",
        type=str,
        help="Path to INPUT_manifest.json file",
    )
    run_parser.add_argument(
        "--env", "-e",
        type=str,
        default=None,
        help="Override environment file (e.g., env_withoutVar.json)",
    )
    run_parser.add_argument(
        "--case-id", "-c",
        type=int,
        default=1,
        help="Case identifier (default: 1)",
    )
    run_parser.add_argument(
        "--info", "-i",
        type=str,
        default="",
        help="Description/info string for the simulation",
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    run_parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate and display plots after simulation",
    )
    run_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files (useful for quick tests)",
    )
    run_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar during simulation",
    )
    run_parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for ODE solver (default: 1e-5)",
    )
    run_parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for ODE solver (default: 1e-9)",
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a manifest file",
    )
    info_parser.add_argument(
        "manifest",
        type=str,
        help="Path to INPUT_manifest.json file",
    )
    
    # Init command (create workspace template)
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new workspace with template files",
    )
    init_parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=".",
        help="Directory to initialize (default: current directory)",
    )
    init_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing files",
    )
    
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    from .Simu import run_a_simulation
    
    manifest_path = Path(args.manifest)
    
    # Handle relative path - check current directory first
    if not manifest_path.is_absolute():
        if not manifest_path.exists():
            # Try current working directory
            cwd_path = Path.cwd() / manifest_path
            if cwd_path.exists():
                manifest_path = cwd_path
    
    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
        print(f"  Current directory: {Path.cwd()}", file=sys.stderr)
        return 1
    
    try:
        results = run_a_simulation(
            manifest_path=manifest_path,
            case_id=args.case_id,
            info_string=args.info,
            results_dir=args.output_dir,
            save_results=not args.no_save,
            show_progress=not args.no_progress,
            plot_results=args.plot,
            rtol=args.rtol,
            atol=args.atol,
            env_file=args.env,
        )
        
        if results.success:
            print(f"\n✓ Simulation completed successfully!")
            return 0
        else:
            print(f"\n✗ Simulation completed with issues: {results.message}", file=sys.stderr)
            return 1
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command - show manifest information."""
    from .params import load_manifest, load_json
    
    manifest_path = Path(args.manifest)
    
    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1
    
    try:
        manifest = load_manifest(manifest_path)
        workspace_path = manifest_path.parent
        
        print("\n" + "=" * 60)
        print(" " * 15 + "MANIFEST INFORMATION")
        print("=" * 60)
        print(f"\n  Manifest: {manifest_path}")
        print(f"  Workspace: {workspace_path}")
        print(f"\n  Files:")
        
        for key, filename in manifest.files.items():
            file_path = workspace_path / filename
            status = "✓" if file_path.exists() else "✗ (missing)"
            print(f"    {key}: {filename} {status}")
        
        # Try to load and show additional info
        if "lineSys" in manifest.files:
            try:
                offsys_data = load_json(workspace_path / manifest.files["lineSys"])
                print(f"\n  System:")
                print(f"    Bodies: {offsys_data.get('nbod', 'N/A')}")
                print(f"    DoF: {offsys_data.get('nDoF', 'N/A')}")
                print(f"    Anchor lines: {offsys_data.get('nAnchorLine', 'N/A')}")
                print(f"    Shared lines: {offsys_data.get('nSharedLine', 'N/A')}")
            except Exception:
                pass
        
        if "simu" in manifest.files:
            try:
                from .params import load_simu_config
                simu = load_simu_config(workspace_path / manifest.files["simu"])
                print(f"\n  Simulation:")
                print(f"    Static: {'enabled' if simu.static_enabled else 'disabled'}")
                print(f"    Dynamic: {'enabled' if simu.dynamic_enabled else 'disabled'}")
                if simu.dynamic_simu:
                    print(f"    Time: {simu.dynamic_simu.tStart}s - {simu.dynamic_simu.tEnd}s")
                    print(f"    dt: {simu.dynamic_simu.dt}s")
            except Exception:
                pass
        
        print("\n" + "=" * 60)
        return 0
        
    except Exception as e:
        print(f"Error reading manifest: {e}", file=sys.stderr)
        return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Execute the init command - create workspace template."""
    import json
    
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInitializing workspace in: {directory.absolute()}")
    
    # Template manifest
    manifest = {
        "workspace_path": ".",
        "files": {
            "lineSys": "system.json",
            "lineType": "line_types.json",
            "env": "env.json",
            "simu": "simu_config.json",
        }
    }
    
    # Template simulation config
    simu_config = {
        "simulator": "offarmpy",
        "static_simu": {
            "enabled": True,
            "parameters": {}
        },
        "dynamic_simu": {
            "enabled": True,
            "numerical_method": {"solver": "RK45"},
            "time_settings": {
                "tStart": 0.0,
                "tEnd": 100.0,
                "dt": 0.1
            }
        }
    }
    
    # Template environment
    env_template = {
        "current": {
            "vel": [0.5],
            "zlevel": [0.0],
            "wakeRatio": 0.9,
            "propDir": 0.0
        },
        "wave": {
            "specType": "Jonswap",
            "Hs": 2.0,
            "Tp": 8.0,
            "gamma": 3.3,
            "propDir": 0.0,
            "dfreq": 0.01,
            "domega": 0.01
        }
    }
    
    files_to_create = [
        ("INPUT_manifest.json", manifest),
        ("simu_config.json", simu_config),
        ("env.json", env_template),
    ]
    
    for filename, content in files_to_create:
        file_path = directory / filename
        if file_path.exists() and not args.force:
            print(f"  Skipping {filename} (exists, use --force to overwrite)")
        else:
            with open(file_path, "w") as f:
                json.dump(content, f, indent=2)
            print(f"  Created {filename}")
    
    # Create diy_configs.py template
    diy_template = '''"""
DIY Config Function for Offarmpy.

Modify this file to customize configurations at runtime.
"""

import numpy as np


def diy_configs(configs, offsys, line_types, env):
    """
    User-defined configuration modifications.
    
    Args:
        configs: Full Configs object (for reference)
        offsys: OffSys object to modify
        line_types: LineTypes object to modify
        env: Env object to modify
        
    Returns:
        Tuple of (offsys, line_types, env)
    """
    print("  Running DIY configs...")
    
    # Add your custom configurations here
    # Example: modify anchor positions, line types, etc.
    
    return offsys, line_types, env
'''
    
    diy_path = directory / "diy_configs.py"
    if diy_path.exists() and not args.force:
        print(f"  Skipping diy_configs.py (exists)")
    else:
        with open(diy_path, "w") as f:
            f.write(diy_template)
        print(f"  Created diy_configs.py")
    
    print(f"\n✓ Workspace initialized!")
    print(f"\nNext steps:")
    print(f"  1. Add your system configuration (system.json)")
    print(f"  2. Add line types configuration (line_types.json)")
    print(f"  3. Modify env.json and simu_config.json as needed")
    print(f"  4. Run: offarm run {directory}/INPUT_manifest.json")
    
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "init":
        return cmd_init(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
