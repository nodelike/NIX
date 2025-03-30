import os
import subprocess
import sys

def setup_environment():
    """
    Set up the environment for the AlphaZero Trader project.
    This will:
    1. Create or update the virtual environment
    2. Install dependencies from requirements.txt
    3. Create necessary directories if they don't exist
    """
    # Print starting message
    print("Setting up environment for AlphaZero Trader...")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Using Python {python_version}")
    
    if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor < 9):
        print("Warning: This project is recommended to use Python 3.9 or newer.")
    
    # Create directories if they don't exist
    directories = ['data', 'models', 'logs', 'instance/flask_session']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("Creating .env file from .env.example...")
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as example_file:
                with open('.env', 'w') as env_file:
                    env_file.write(example_file.read())
            print(".env file created. Please edit it with your settings.")
        else:
            print("Warning: .env.example not found. Please create a .env file manually.")
    
    # Install requirements
    print("\nInstalling dependencies from requirements.txt...")
    try:
        # Use --upgrade-strategy eager to force reinstall of all packages
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade-strategy", "eager"], check=True)
        print("Dependencies successfully installed!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("Try running: pip install -r requirements.txt manually")
        return False
    
    print("\nEnvironment setup complete! You can now run the application with:")
    print("  python app.py")
    
    return True

if __name__ == "__main__":
    setup_environment() 