from flask import Flask
from flask_assets import Environment, Bundle
from flask_session import Session
import os
import secrets
import logging

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate a strong random key for session encryption
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', secrets.token_hex(32)),
        DATABASE=os.path.join(app.instance_path, 'indiavix.sqlite'),
        SESSION_TYPE=os.environ.get('SESSION_TYPE', 'filesystem'),
        SESSION_PERMANENT=os.environ.get('SESSION_PERMANENT', 'true').lower() == 'true',
        PERMANENT_SESSION_LIFETIME=int(os.environ.get('SESSION_LIFETIME', 1800)),  # 30 minutes
        SESSION_FILE_DIR=os.environ.get('SESSION_FILE_DIR', os.path.join(app.instance_path, 'flask_session')),
        SESSION_USE_SIGNER=os.environ.get('SESSION_USE_SIGNER', 'true').lower() == 'true',
        SESSION_FILE_THRESHOLD=int(os.environ.get('SESSION_FILE_THRESHOLD', 100)),
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
        
    # Ensure the session directory exists
    if app.config['SESSION_TYPE'] == 'filesystem':
        try:
            os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
        except OSError as e:
            app.logger.error(f"Could not create session directory: {e}")

    # Initialize Flask-Session
    Session(app)

    # Initialize Flask-Assets
    assets = Environment(app)
    css = Bundle(
        'css/tailwind.css',
        filters='cssmin',
        output='gen/packed.css'
    )
    js = Bundle(
        'js/main.js',
        filters='jsmin',
        output='gen/packed.js'
    )
    assets.register('css_all', css)
    assets.register('js_all', js)

    # Register blueprints
    from app.routes import main, data, training, backtesting, prediction, settings
    app.register_blueprint(main.bp)
    app.register_blueprint(data.bp)
    app.register_blueprint(training.bp)
    app.register_blueprint(backtesting.bp)
    app.register_blueprint(prediction.bp)
    app.register_blueprint(settings.bp)

    # Make the index route accessible at the root URL
    app.add_url_rule('/', endpoint='index')

    # Log successful initialization
    app.logger.info("Application initialized successfully")
    
    return app
