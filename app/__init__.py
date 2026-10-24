import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS  # 1. Import CORS
from sqlalchemy import inspect, text
from dotenv import load_dotenv

# 1. Load environment variables from .env file first
load_dotenv()

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    
    # 2. Enable CORS so your React frontend (http://localhost:5173) can talk to Flask[cite: 4]
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # 2. Fetch the DATABASE_URL (Remove the fallback 'localhost' string so it fails loudly if missing)[cite: 4]
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError(" ERROR: DATABASE_URL is not set in environment variables or .env file!")

    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    with app.app_context():
        from app.model import VideoJob
        db.create_all()
        job_columns = {
            column["name"] for column in inspect(db.engine).get_columns("video_jobs")
        }
        with db.engine.begin() as connection:
            if "progress" not in job_columns:
                connection.execute(
                    text(
                        "ALTER TABLE video_jobs "
                        "ADD COLUMN progress INTEGER NOT NULL DEFAULT 0"
                    )
                )
            if "progress_message" not in job_columns:
                connection.execute(
                    text(
                        "ALTER TABLE video_jobs "
                        "ADD COLUMN progress_message VARCHAR(200)"
                    )
                )

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app