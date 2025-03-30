import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Base configuration class"""
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_for_development_only')
    PORT = int(os.environ.get('PORT', 8080))
    
    # Session Configuration
    SESSION_TYPE = os.environ.get('SESSION_TYPE', 'filesystem')
    SESSION_PERMANENT = os.environ.get('SESSION_PERMANENT', 'true').lower() == 'true'
    PERMANENT_SESSION_LIFETIME = int(os.environ.get('SESSION_LIFETIME', 1800))  # 30 minutes
    SESSION_USE_SIGNER = os.environ.get('SESSION_USE_SIGNER', 'true').lower() == 'true'
    SESSION_FILE_THRESHOLD = int(os.environ.get('SESSION_FILE_THRESHOLD', 100))

    # Application Directories
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.environ.get('DATA_DIR', os.path.join(APP_ROOT, 'data'))
    MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(APP_ROOT, 'models'))
    
    # File Names
    MODEL_FILE = os.environ.get('MODEL_FILE', 'alphazero_model')
    
    # Trading Parameters
    WINDOW_SIZE = int(os.environ.get('WINDOW_SIZE', 10))
    TRADE_TIME = os.environ.get('TRADE_TIME', '9:15')
    DEFAULT_LOT_SIZE = int(os.environ.get('DEFAULT_LOT_SIZE', 50))
    DEFAULT_INITIAL_CAPITAL = int(os.environ.get('DEFAULT_INITIAL_CAPITAL', 100000))
    
    # Ensure required directories exist
    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HOST = '127.0.0.1'  # Localhost only for development


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HOST = '0.0.0.0'  # All interfaces for production
    
    # Use environment-provided values or secure defaults
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Will raise error if not set
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    PREFERRED_URL_SCHEME = 'https'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    HOST = '127.0.0.1'
    
    # Use in-memory session for testing
    SESSION_TYPE = 'filesystem'
    
    # Use test directories
    DATA_DIR = os.path.join(Config.APP_ROOT, 'tests', 'data')
    MODEL_DIR = os.path.join(Config.APP_ROOT, 'tests', 'models')


# Set the configuration based on environment
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig, 
    'testing': TestingConfig
}

# Get the current environment or default to development
ENV = os.environ.get('FLASK_ENV', 'development').lower()

# Get the appropriate configuration class
AppConfig = config_map.get(ENV, DevelopmentConfig)

# Ensure directories exist
AppConfig.ensure_directories() 