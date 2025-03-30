#!/usr/bin/env python
# debug_config.py - Debug script to test config loading

import yaml
import os
import boto3

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
            print("AWS configuration:")
            print(f"  Region: {config['aws']['region']}")
            print(f"  Profile: {config['aws']['profile']}")
            return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return None

def test_aws_connection(config):
    """Test AWS connection using the loaded configuration"""
    if not config:
        return False
    
    try:
        # Create a boto3 session with the specified profile
        profile = config['aws']['profile']
        region = config['aws']['region']
        
        print(f"Attempting AWS connection with profile '{profile}' and region '{region}'")
        
        session = boto3.Session(profile_name=profile, region_name=region)
        sts = session.client('sts')
        
        # Get caller identity to verify credentials
        identity = sts.get_caller_identity()
        print("AWS connection successful!")
        print(f"Connected as: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"AWS connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    config = load_config()
    test_aws_connection(config)