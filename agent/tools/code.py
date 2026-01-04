"""
Code Modification Tools
=======================

Tools for the agent to read and modify Python source code.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
HEURISTICS_ROOT = PROJECT_ROOT / "model_heuristics"
SCRIPTS_DIR = HEURISTICS_ROOT / "scripts"

# Files the agent can modify
ALLOWED_FILES = [
    "model_heuristics/scripts/value_heuristic_model.py",
    "model_heuristics/scripts/factor_vs_spy_analysis.py",
    "model_heuristics/scripts/combined_factor_strategy.py",
    "model_heuristics/scripts/analyze_forward_bias_predictors.py",
    "model_heuristics/scripts/value_vs_spy_heuristic.py",
]


def list_model_scripts() -> Dict[str, Any]:
    """
    List all available model scripts that can be modified.
    """
    scripts = []
    for script_path in SCRIPTS_DIR.glob("*.py"):
        rel_path = str(script_path.relative_to(PROJECT_ROOT))
        with open(script_path, 'r') as f:
            content = f.read()
            lines = content.count('\n') + 1
            # Get docstring
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            docstring = docstring_match.group(1).strip()[:200] if docstring_match else ""
        
        scripts.append({
            "path": rel_path,
            "name": script_path.name,
            "lines": lines,
            "description": docstring,
        })
    
    return {
        "scripts_dir": str(SCRIPTS_DIR.relative_to(PROJECT_ROOT)),
        "scripts": scripts,
    }


def read_file(file_path: str, start_line: int = 1, end_line: Optional[int] = None) -> Dict[str, Any]:
    """
    Read a file's contents.
    
    Args:
        file_path: Path relative to project root
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (optional)
        
    Returns:
        Dict with file contents
    """
    full_path = PROJECT_ROOT / file_path
    
    if not full_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line) if end_line else total_lines
        
        selected_lines = lines[start_idx:end_idx]
        
        # Add line numbers
        numbered_content = ""
        for i, line in enumerate(selected_lines, start=start_idx + 1):
            numbered_content += f"{i:4d} | {line}"
        
        return {
            "file_path": file_path,
            "total_lines": total_lines,
            "showing_lines": f"{start_idx + 1} to {end_idx}",
            "content": numbered_content,
        }
    except Exception as e:
        return {"error": str(e)}


def write_file(file_path: str, content: str, backup: bool = True) -> Dict[str, Any]:
    """
    Write content to a file (overwrite).
    
    Args:
        file_path: Path relative to project root
        content: New file content
        backup: Create backup before writing
        
    Returns:
        Success/error message
    """
    # Security check
    if file_path not in ALLOWED_FILES:
        return {
            "error": f"Cannot modify {file_path}. Allowed files: {ALLOWED_FILES}"
        }
    
    full_path = PROJECT_ROOT / file_path
    
    try:
        # Create backup
        if backup and full_path.exists():
            backup_path = full_path.with_suffix('.py.bak')
            with open(full_path, 'r') as f:
                original = f.read()
            with open(backup_path, 'w') as f:
                f.write(original)
        
        # Write new content
        with open(full_path, 'w') as f:
            f.write(content)
        
        return {
            "success": True,
            "file_path": file_path,
            "lines_written": content.count('\n') + 1,
            "backup_created": backup,
        }
    except Exception as e:
        return {"error": str(e)}


def search_replace_in_file(
    file_path: str,
    old_text: str,
    new_text: str,
) -> Dict[str, Any]:
    """
    Search and replace text in a file.
    
    Args:
        file_path: Path relative to project root
        old_text: Text to find
        new_text: Replacement text
        
    Returns:
        Success/error with number of replacements
    """
    # Security check
    if file_path not in ALLOWED_FILES:
        return {
            "error": f"Cannot modify {file_path}. Allowed files: {ALLOWED_FILES}"
        }
    
    full_path = PROJECT_ROOT / file_path
    
    if not full_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Count occurrences
        count = content.count(old_text)
        
        if count == 0:
            return {
                "error": "Text not found in file",
                "searched_for": old_text[:100] + "..." if len(old_text) > 100 else old_text,
            }
        
        # Create backup
        backup_path = full_path.with_suffix('.py.bak')
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Replace
        new_content = content.replace(old_text, new_text)
        
        with open(full_path, 'w') as f:
            f.write(new_content)
        
        return {
            "success": True,
            "file_path": file_path,
            "replacements": count,
            "backup_created": True,
        }
    except Exception as e:
        return {"error": str(e)}


def insert_code_at_line(
    file_path: str,
    line_number: int,
    code: str,
) -> Dict[str, Any]:
    """
    Insert code at a specific line number.
    
    Args:
        file_path: Path relative to project root
        line_number: Line number to insert at (1-indexed)
        code: Code to insert
        
    Returns:
        Success/error message
    """
    # Security check
    if file_path not in ALLOWED_FILES:
        return {
            "error": f"Cannot modify {file_path}. Allowed files: {ALLOWED_FILES}"
        }
    
    full_path = PROJECT_ROOT / file_path
    
    if not full_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        # Create backup
        backup_path = full_path.with_suffix('.py.bak')
        with open(backup_path, 'w') as f:
            f.writelines(lines)
        
        # Insert code
        insert_idx = min(line_number - 1, len(lines))
        code_lines = code.split('\n')
        for i, code_line in enumerate(code_lines):
            lines.insert(insert_idx + i, code_line + '\n')
        
        with open(full_path, 'w') as f:
            f.writelines(lines)
        
        return {
            "success": True,
            "file_path": file_path,
            "inserted_at_line": line_number,
            "lines_inserted": len(code_lines),
        }
    except Exception as e:
        return {"error": str(e)}


def run_script(
    script_path: str,
    args: Optional[List[str]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Run a Python script and capture output.
    
    Args:
        script_path: Path relative to project root
        args: Command line arguments
        timeout: Timeout in seconds
        
    Returns:
        Script output and exit code
    """
    full_path = PROJECT_ROOT / script_path
    
    if not full_path.exists():
        return {"error": f"Script not found: {script_path}"}
    
    cmd = ["python", str(full_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        return {
            "script": script_path,
            "exit_code": result.returncode,
            "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Script timed out after {timeout} seconds"}
    except Exception as e:
        return {"error": str(e)}


def update_signal_weights(
    file_path: str,
    weights: Dict[str, float],
    threshold: float = 0.0,
    hold_period: int = 0,
) -> Dict[str, Any]:
    """
    Update signal weights in a heuristic model file.
    
    This is a specialized function that updates the compute_heuristic_signal
    or similar methods in the model files.
    
    Args:
        file_path: Path to the model file
        weights: Dict of feature name to weight
        threshold: Signal threshold
        hold_period: Holding period in days
        
    Returns:
        Success/error message
    """
    # Security check
    if file_path not in ALLOWED_FILES:
        return {
            "error": f"Cannot modify {file_path}. Allowed files: {ALLOWED_FILES}"
        }
    
    full_path = PROJECT_ROOT / file_path
    
    if not full_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_path = full_path.with_suffix('.py.bak')
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Generate new signal code
        weight_code = "        # === AGENT-OPTIMIZED SIGNAL WEIGHTS ===\n"
        weight_code += "        # Updated by optimization agent\n"
        weight_code += f"        # Threshold: {threshold}, Hold period: {hold_period}\n"
        weight_code += "        signal = pd.Series(0.0, index=features.index)\n"
        weight_code += "        weights_used = 0\n\n"
        
        for feature, weight in weights.items():
            sign = "+" if weight > 0 else ""
            weight_code += f"        # {feature}: {sign}{weight:.3f}\n"
            weight_code += f"        if '{feature}' in features.columns:\n"
            weight_code += f"            feat_norm = (features['{feature}'] - features['{feature}'].rolling(252).mean()) / features['{feature}'].rolling(252).std()\n"
            weight_code += f"            feat_norm = feat_norm.clip(-3, 3) / 3\n"
            weight_code += f"            signal += {weight:.4f} * feat_norm.fillna(0)\n"
            weight_code += f"            weights_used += abs({weight:.4f})\n\n"
        
        weight_code += "        # Normalize\n"
        weight_code += "        if weights_used > 0:\n"
        weight_code += "            signal = signal / weights_used\n"
        weight_code += "        # === END AGENT-OPTIMIZED WEIGHTS ===\n"
        
        # Find and replace the signal computation section
        # Look for existing agent-optimized section first
        agent_section_pattern = r'# === AGENT-OPTIMIZED SIGNAL WEIGHTS ===.*?# === END AGENT-OPTIMIZED WEIGHTS ==='
        
        if re.search(agent_section_pattern, content, re.DOTALL):
            # Replace existing agent section
            new_content = re.sub(agent_section_pattern, weight_code.strip(), content, flags=re.DOTALL)
        else:
            # Can't auto-insert without knowing the right location
            return {
                "error": "Could not find insertion point. Use search_replace_in_file or insert_code_at_line instead.",
                "suggested_code": weight_code,
            }
        
        with open(full_path, 'w') as f:
            f.write(new_content)
        
        return {
            "success": True,
            "file_path": file_path,
            "weights_updated": weights,
            "threshold": threshold,
            "hold_period": hold_period,
            "backup_created": True,
        }
    except Exception as e:
        return {"error": str(e)}


def lint_file(file_path: str) -> Dict[str, Any]:
    """
    Run Python linter on a file to check for errors.
    
    Args:
        file_path: Path relative to project root
        
    Returns:
        Linting results
    """
    full_path = PROJECT_ROOT / file_path
    
    if not full_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        # Try python -m py_compile for syntax check
        result = subprocess.run(
            ["python", "-m", "py_compile", str(full_path)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            return {
                "file_path": file_path,
                "valid": True,
                "message": "No syntax errors found",
            }
        else:
            return {
                "file_path": file_path,
                "valid": False,
                "errors": result.stderr,
            }
    except Exception as e:
        return {"error": str(e)}

