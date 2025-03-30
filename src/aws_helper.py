#!/usr/bin/env python
# aws_helper.py - Helper functions for AWS credentials

import os
import boto3
import yaml
import subprocess
import logging

logger = logging.getLogger(__name__)

def load_config(config_file='../config/hpomapper_config.yaml'):
    """Load configuration from yaml file"""
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config
        else:
            logger.warning(f"Configuration file {config_file} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def get_aws_session(profile_name=None, region_name=None):
    """
    Get AWS session with proper credential handling
    
    This function tries multiple approaches to get valid AWS credentials:
    1. Use environment variables if set
    2. Use the specified profile
    3. Fall back to default credentials if available
    """
    # Load configuration
    config = load_config()
    
    # Try environment variables first
    if os.environ.get('AWS_PROFILE'):
        env_profile = os.environ.get('AWS_PROFILE')
        logger.info(f"Using AWS profile from environment: {env_profile}")
        profile_name = env_profile
    
    # If profile wasn't specified in args, try to get from config
    if profile_name is None and 'aws' in config and 'profile' in config['aws']:
        profile_name = config['aws']['profile']
        logger.info(f"Using AWS profile from config: {profile_name}")
    
    # If region wasn't specified in args, try to get from config or environment
    if region_name is None:
        if os.environ.get('AWS_DEFAULT_REGION'):
            region_name = os.environ.get('AWS_DEFAULT_REGION')
            logger.info(f"Using AWS region from environment: {region_name}")
        elif 'aws' in config and 'region' in config['aws']:
            region_name = config['aws']['region']
            logger.info(f"Using AWS region from config: {region_name}")
    
    # Create the session
    try:
        if profile_name:
            logger.info(f"Creating AWS session with profile: {profile_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
        else:
            logger.info("Creating default AWS session")
            session = boto3.Session(region_name=region_name)
        
        # Verify the session has valid credentials
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"AWS session created successfully. Using identity: {identity['Arn']}")
        
        return session
    except Exception as e:
        logger.error(f"Failed to create AWS session: {str(e)}")
        
        # If using a profile and it failed, try to login
        if profile_name:
            logger.info(f"Attempting to login with SSO for profile {profile_name}")
            try:
                subprocess.run(['aws', 'sso', 'login', '--profile', profile_name], check=True)
                
                # Try creating the session again
                logger.info("Retrying session creation after SSO login")
                return boto3.Session(profile_name=profile_name, region_name=region_name)
            except Exception as login_err:
                logger.error(f"SSO login failed: {str(login_err)}")
        
        # Last resort: try default session
        logger.info("Falling back to default AWS session")
        return boto3.Session(region_name=region_name)

def get_bedrock_client(session=None, profile_name=None, region_name=None):
    """Get a Bedrock client with valid AWS credentials"""
    if session is None:
        session = get_aws_session(profile_name, region_name)
    
    try:
        client = session.client('bedrock-runtime')
        logger.info("Created Bedrock client successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create Bedrock client: {str(e)}")
        raise