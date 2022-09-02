from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()



class Generation(db.Model):
    __tablename__ = 'generations'

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    created_at = db.Column(db.DateTime(True), server_default=db.FetchedValue())
    updated_at = db.Column(db.DateTime(True), server_default=db.FetchedValue())
    opt = db.Column(db.JSON)
    request_headers = db.Column(db.Text)
    request_raw = db.Column(db.Text)
    image_data = db.Column(db.Text)
