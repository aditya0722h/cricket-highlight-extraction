from app import db
from datetime import datetime

class VideoJob(db.Model):
    __tablename__ = 'video_jobs'

    id = db.Column(db.Integer, primary_key=True)
    job_type = db.Column(db.String(50), nullable=False, default='replay') # 'replay' or 'live'
    source_path = db.Column(db.String(500), nullable=False) # File path or stream URL
    status = db.Column(db.String(50), default='queued') # queued, processing, completed, failed
    progress = db.Column(db.Integer, nullable=False, default=0)
    progress_message = db.Column(db.String(200), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "job_type": self.job_type,
            "source_path": self.source_path,
            "status": self.status,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat()
        }

class TrainingLog(db.Model):
    __tablename__ = 'training_logs'

    id = db.Column(db.Integer, primary_key=True)
    match_name = db.Column(db.String(200), nullable=False)
    ground_truth_path = db.Column(db.String(500), nullable=False)
    weights_version = db.Column(db.String(100), nullable=False, default='v1_initial')
    status = db.Column(db.String(50), default='training') # training, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "match_name": self.match_name,
            "weights_version": self.weights_version,
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }