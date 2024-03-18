
import sqlalchemy as sa
import sqlalchemy.orm as so
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
from hashlib import md5
from typing import Optional
from app import db, login

class Dealer(UserMixin, db.Model):
    id: so.Mapped[int] = sa.Column(sa.Integer, primary_key=True, nullable=False)
    name: so.Mapped[str] = sa.Column(sa.String(256), index=True, unique=True)
    password_hash: so.Mapped[Optional[str]] = sa.Column(sa.String(256))

    def __repr__(self):
        return '<Dealer {}>'.format(self.name)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login.user_loader
def load_user(id):
    return db.session.get(Dealer, int(id))

"""
class Transaction(UserMixin, db.Model):
    id: so.Mapped[int] = sa.Column(sa.Integer, primary_key=True, nullable=False)
    amount: so.Mapped[Optional[float]] = sa.Column(sa.Float)
    kind: so.Mapped[Optional[int]] = sa.Column(sa.Integer)
    step: so.Mapped[Optional[int]] = sa.Column(sa.Integer)
    nameOrg: so.Mapped[Optional[str]] = sa.Column(sa.String(50))
    nameDest: so.Mapped[Optional[str]] = sa.Column(sa.String(50))
    oldbalanceOrg: so.Mapped[Optional[float]] = sa.Column(sa.Float)
    newbalanceOrg: so.Mapped[Optional[float]] = sa.Column(sa.Float)
    oldbalanceDest: so.Mapped[Optional[float]] = sa.Column(sa.Float)
    newbalanceDest: so.Mapped[Optional[float]] = sa.Column(sa.Float)
    timestamp: so.Mapped[datetime] = sa.Column(sa.DateTime, index=True, default=lambda: datetime.now(timezone.utc))
    fraud: so.Mapped[Optional[bool]] = sa.Column(sa.Boolean)
    employee_id: so.Mapped[int] = sa.Column(sa.ForeignKey(Employee.id), index=True)
    employee: so.Mapped[Employee] = so.relationship("Employee", back_populates='transactions')

    def __repr__(self):
        return '<Transaction(id={}, amount={}, kind={}, step={}, nameOrg={}, nameDest={}, oldbalanceOrg={}, newbalanceOrg={}, oldbalanceDest={}, newbalanceDest={}, timestamp={}, fraud={})>'.format(
            self.id, self.amount, self.kind, self.step, self.nameOrg, self.nameDest, self.oldbalanceOrg, self.newbalanceOrg,
            self.oldbalanceDest, self.newbalanceDest, self.timestamp, self.fraud
        )
"""