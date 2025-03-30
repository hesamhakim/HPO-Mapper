#!/usr/bin/env python
# phenoscope_runner.py - Wrapper script to handle AWS SSO and run PhenoScope

import os
import sys
import subprocess
import yaml
import boto3
import json
from datetime import datetime, timedelta
import time

def load_config():
    """Load configuration from phenoscope_config.yaml file"""
    config_file = 'phenoscope_config.yaml'
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found!")
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Successfully loaded configuration from {config_file}")
            return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return None

def check_sso_credentials(profile):
    """Check if AWS SSO credentials are valid and not expired"""
    try:
        # Create a session with the profile
        session = boto3.Session(profile_name=profile)
        sts = session.client('sts')
        
        # Try to get caller identity
        sts.get_caller_identity()
        print(f"AWS SSO credentials for profile '{profile}' are valid")
        return True
    except Exception as e:
        print(f"AWS SSO credentials check failed: {str(e)}")
        return False

def login_sso(profile):
    """Log in to AWS SSO with the specified profile"""
    try:
        print(f"Logging in to AWS SSO with profile '{profile}'...")
        result = subprocess.run(['aws', 'sso', 'login', '--profile', profile], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("AWS SSO login successful")
            return True
        else:
            print(f"AWS SSO login failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error during AWS SSO login: {str(e)}")
        return False

def run_phenoscope(args):
    """Run the main PhenoScope script with the provided arguments"""
    try:
        cmd = [sys.executable, 'phenoscope.py'] + args
        print(f"Running PhenoScope with command: {' '.join(cmd)}")
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error running PhenoScope: {str(e)}")

def main():
    # Load configuration
    config = load_config()
    if not config:
        sys.exit(1)
    
    # Get AWS profile from config
    profile = config['aws'].get('profile', 'plm-dev')
    
    # Check and refresh SSO credentials if needed
    if not check_sso_credentials(profile):
        print("AWS SSO credentials are invalid or expired")
        if not login_sso(profile):
            print("Failed to log in to AWS SSO. Please try manually with:")
            print(f"  aws sso login --profile {profile}")
            sys.exit(1)
    
    # Get command line arguments for PhenoScope
    phenoscope_args = sys.argv[1:]
    
    # Run PhenoScope
    run_phenoscope(phenoscope_args)

if __name__ == "__main__":
    main()