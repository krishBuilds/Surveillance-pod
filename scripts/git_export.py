#!/usr/bin/env python3
"""
Git Export and Repository Management Script
Provides utilities for git operations and repository export
"""

import os
import subprocess
import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

def run_git_command(command, cwd=None):
    """Run a git command and return the result"""
    try:
        if cwd is None:
            cwd = os.getcwd()
        
        result = subprocess.run(
            command.split() if isinstance(command, str) else command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def setup_git_config():
    """Setup basic git configuration if not already set"""
    print("Setting up git configuration...")
    
    # Check if user.name is set
    success, name = run_git_command("git config --global user.name")
    if not success or not name:
        print("Setting default git user name...")
        run_git_command("git config --global user.name 'InternVideo2.5 Surveillance'")
    
    # Check if user.email is set
    success, email = run_git_command("git config --global user.email")
    if not success or not email:
        print("Setting default git user email...")
        run_git_command("git config --global user.email 'internvideo@surveillance.local'")
    
    print("Git configuration complete")

def check_git_status():
    """Check the current git repository status"""
    print("=== Git Repository Status ===")
    
    # Check if we're in a git repository
    success, _ = run_git_command("git rev-parse --git-dir")
    if not success:
        print("❌ Not in a git repository")
        return False
    
    # Get repository info
    success, branch = run_git_command("git branch --show-current")
    if success:
        print(f"📍 Current branch: {branch}")
    
    # Check status
    success, status = run_git_command("git status --porcelain")
    if success:
        if status:
            print(f"📝 Modified files: {len(status.splitlines())} files")
            print("Changes to commit:")
            for line in status.splitlines()[:10]:  # Show first 10 files
                print(f"   {line}")
            if len(status.splitlines()) > 10:
                print(f"   ... and {len(status.splitlines()) - 10} more files")
        else:
            print("✅ Working directory clean")
    
    # Check commits
    success, log = run_git_command("git log --oneline -5")
    if success and log:
        print("📚 Recent commits:")
        for line in log.splitlines():
            print(f"   {line}")
    else:
        print("📚 No commits yet")
    
    return True

def add_files_to_git():
    """Add files to git staging area"""
    print("=== Adding Files to Git ===")
    
    # Add all files except those in .gitignore
    success, output = run_git_command("git add .")
    if success:
        print("✅ Files added to staging area")
    else:
        print(f"❌ Error adding files: {output}")
        return False
    
    # Show what was staged
    success, status = run_git_command("git status --cached --porcelain")
    if success and status:
        print(f"📦 Staged files ({len(status.splitlines())} files):")
        for line in status.splitlines()[:15]:  # Show first 15 files
            print(f"   {line}")
        if len(status.splitlines()) > 15:
            print(f"   ... and {len(status.splitlines()) - 15} more files")
    
    return True

def commit_changes(message=None):
    """Commit changes to git"""
    if message is None:
        message = f"InternVideo2.5 Surveillance System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    print(f"=== Committing Changes ===")
    print(f"📝 Commit message: {message}")
    
    success, output = run_git_command(f'git commit -m "{message}"')
    if success:
        print("✅ Changes committed successfully")
        return True
    else:
        if "nothing to commit" in output:
            print("ℹ️ No changes to commit")
            return True
        else:
            print(f"❌ Error committing: {output}")
            return False

def setup_remote_origin(repo_url):
    """Setup remote origin for the repository"""
    print(f"=== Setting up remote origin ===")
    print(f"🔗 Repository URL: {repo_url}")
    
    # Check if origin already exists
    success, remotes = run_git_command("git remote -v")
    if success and "origin" in remotes:
        print("🔄 Removing existing origin...")
        run_git_command("git remote remove origin")
    
    # Add new origin
    success, output = run_git_command(f"git remote add origin {repo_url}")
    if success:
        print("✅ Remote origin added successfully")
        return True
    else:
        print(f"❌ Error adding remote: {output}")
        return False

def push_to_remote(branch="main"):
    """Push changes to remote repository"""
    print(f"=== Pushing to Remote Repository ===")
    print(f"🚀 Pushing branch: {branch}")
    
    # Create and switch to main branch if we're on master
    success, current_branch = run_git_command("git branch --show-current")
    if success and current_branch == "master":
        print("🔄 Switching from master to main branch...")
        run_git_command("git branch -m main")
        branch = "main"
    
    # Push to remote
    success, output = run_git_command(f"git push -u origin {branch}")
    if success:
        print("✅ Successfully pushed to remote repository")
        return True
    else:
        print(f"❌ Error pushing to remote: {output}")
        if "Authentication failed" in output:
            print("💡 Make sure you have proper authentication set up (SSH keys or personal access token)")
        elif "repository not found" in output:
            print("💡 Make sure the repository exists and the URL is correct")
        return False

def create_export_summary():
    """Create a summary of the exported project"""
    summary = {
        "project_name": "InternVideo2.5 Surveillance System",
        "export_date": datetime.now().isoformat(),
        "description": "Advanced video understanding system with InternVideo2.5, 8-bit quantization, and temporal analysis",
        "features": [
            "InternVideo2.5 Chat 8B model with 8-bit quantization",
            "160-frame capacity with RTX 4090 optimization", 
            "Web interface with streaming chat responses",
            "Temporal analysis and video snippet generation",
            "Flash Attention 2.8.3 integration",
            "Singleton model architecture for memory efficiency"
        ],
        "key_files": [
            "core/model_manager.py - Core model management with quantization",
            "main.py - CLI interface for video analysis", 
            "main_enhanced.py - Enhanced model manager with temporal analysis",
            "webapp/app.py - Flask web interface",
            "scripts/download_model.py - Model download utility",
            "config.yaml - System configuration",
            "requirements.txt - Python dependencies"
        ],
        "excluded_dirs": [
            "models/ - Large model files (excluded from git)",
            "inputs/ - Input video files (excluded from git)", 
            "outputs/ - Generated outputs (excluded from git)"
        ]
    }
    
    with open("PROJECT_EXPORT_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("📋 Created PROJECT_EXPORT_SUMMARY.json")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Git Export and Repository Management")
    parser.add_argument("--repo-url", help="Git repository URL for remote origin")
    parser.add_argument("--commit-message", help="Custom commit message")
    parser.add_argument("--branch", default="main", help="Branch to push to (default: main)")
    parser.add_argument("--status-only", action="store_true", help="Only show git status")
    parser.add_argument("--setup-only", action="store_true", help="Only setup git config")
    
    args = parser.parse_args()
    
    print("🚀 InternVideo2.5 Surveillance - Git Export Tool")
    print("=" * 50)
    
    # Setup git configuration
    setup_git_config()
    
    if args.setup_only:
        print("✅ Git setup complete")
        return
    
    # Check git status
    if not check_git_status():
        print("❌ Git repository not found. Run 'git init' first.")
        return
    
    if args.status_only:
        print("✅ Status check complete")
        return
    
    # Create export summary
    summary = create_export_summary()
    
    # Add files to git
    if not add_files_to_git():
        return
    
    # Commit changes
    if not commit_changes(args.commit_message):
        return
    
    # Setup remote and push if URL provided
    if args.repo_url:
        if setup_remote_origin(args.repo_url):
            if push_to_remote(args.branch):
                print("\n🎉 Export to git repository completed successfully!")
                print(f"📍 Repository: {args.repo_url}")
                print(f"🌿 Branch: {args.branch}")
            else:
                print("\n⚠️ Export completed locally but push to remote failed")
        else:
            print("\n⚠️ Export completed locally but remote setup failed")
    else:
        print("\n✅ Local git export completed successfully!")
        print("💡 To push to remote repository, run with --repo-url option")
    
    print("\n📋 Project Summary:")
    print(f"   Features: {len(summary['features'])}")
    print(f"   Key Files: {len(summary['key_files'])}")
    print(f"   Export Date: {summary['export_date']}")

if __name__ == "__main__":
    main()