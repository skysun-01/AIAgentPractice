"""
Simulated email service: FastAPI (REST endpoints) + SQLite/SQLAlchemy (storage) + Pydantic (validation).

You don't need to start it yourself: importing `utils` or `email_tools` calls `ensure_running()`,
which starts it in a background thread if nothing is answering yet. To run it on its own
(and browse the interactive API docs at http://127.0.0.1:8765/docs):

    python email_service.py

Settings (environment variables): EMAIL_API_PORT (default 8765), EMAIL_API_URL (use a server elsewhere),
EMAIL_DB_PATH (default: emails.db next to this file; the evals use their own copy).
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, or_, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

HOST = "127.0.0.1"
PORT = int(os.getenv("EMAIL_API_PORT", "8765"))
BASE_URL = os.getenv("EMAIL_API_URL", f"http://{HOST}:{PORT}").rstrip("/")
DB_PATH = Path(os.getenv("EMAIL_DB_PATH") or Path(__file__).with_name("emails.db"))
MY_ADDRESS = "you@email.com"
SERVICE_NAME = "simulated-email"


# --------------------------------------------------------------------------------------
# Storage (SQLite + SQLAlchemy)
# --------------------------------------------------------------------------------------
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sender: Mapped[str] = mapped_column(String(255))
    recipient: Mapped[str] = mapped_column(String(255))
    subject: Mapped[str] = mapped_column(String(255))
    body: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    read: Mapped[bool] = mapped_column(Boolean, default=False)


# (sender, recipient, subject, body, timestamp, read) — id 1 is the lab's "Happy Hour" email
SEED_EMAILS = [
    ("eric@work.com", MY_ADDRESS, "Happy Hour",
     "We're planning drinks this Friday!", "2025-06-13T04:48:59.096908", False),
    ("boss@email.com", MY_ADDRESS, "Quarterly report",
     "Hi, please send me the Q2 report by Thursday. Let me know if anything is blocking you.",
     "2025-06-13T08:15:12", False),
    ("alice@work.com", MY_ADDRESS, "Design review notes",
     "Here are my notes from today's design review. Can you take a look before tomorrow?",
     "2025-06-12T16:42:30", False),
    ("bob@work.com", MY_ADDRESS, "Lunch tomorrow?",
     "Want to grab lunch at the new taco place at noon?", "2025-06-12T11:05:44", False),
    ("it@company.com", MY_ADDRESS, "Password expires in 3 days",
     "Your password will expire soon. Please update it via the self-service portal.",
     "2025-06-12T09:20:00", False),
    ("you@email.com", "carol@client.com", "Proposal draft",
     "Hi Carol, here is the draft proposal we discussed. Feedback welcome!",
     "2025-06-11T17:45:03", True),
    ("hr@company.com", MY_ADDRESS, "Benefits enrollment closes Friday",
     "Reminder: open enrollment ends this Friday. Review your selections in the HR portal.",
     "2025-06-11T14:00:00", True),
    ("boss@email.com", MY_ADDRESS, "Team offsite",
     "The team offsite is confirmed for June 27th. Agenda to follow.", "2025-06-10T09:30:21", True),
    ("newsletter@techdigest.com", MY_ADDRESS, "This week in AI",
     "Top stories: new agent frameworks, tool-calling benchmarks, and more.",
     "2025-06-09T07:00:00", True),
    ("dana@client.com", MY_ADDRESS, "Invoice #4521",
     "Please find invoice #4521 for May services. Payment is due in 30 days.",
     "2025-06-08T10:10:10", True),
]


def reset_db() -> int:
    """Drop everything and reload the sample inbox (ids restart at 1)."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        db.add_all(
            Email(sender=s, recipient=r, subject=subj, body=b, timestamp=datetime.fromisoformat(ts), read=rd)
            for s, r, subj, b, ts, rd in SEED_EMAILS
        )
        db.commit()
    return len(SEED_EMAILS)


def init_db() -> None:
    """Create and seed the database on first use; keep existing data otherwise."""
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        if db.scalar(select(Email.id).limit(1)) is None:
            reset_db()


def get_db():
    with SessionLocal() as db:
        yield db


# --------------------------------------------------------------------------------------
# Validation (Pydantic)
# --------------------------------------------------------------------------------------
class EmailIn(BaseModel):
    recipient: str
    subject: str
    body: str


class EmailOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sender: str
    recipient: str
    subject: str
    body: str
    timestamp: datetime
    read: bool


# --------------------------------------------------------------------------------------
# REST endpoints (FastAPI)
# --------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Simulated Email Service", lifespan=lifespan)


def _get_or_404(db: Session, email_id: int) -> Email:
    email = db.get(Email, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"Email {email_id} not found")
    return email


def _newest_first(query):
    return query.order_by(Email.timestamp.desc(), Email.id.desc())


@app.get("/health")
def health():
    return {"service": SERVICE_NAME, "status": "ok"}


@app.post("/send", response_model=EmailOut)
def send_email(email: EmailIn, db: Session = Depends(get_db)):
    # Simulated: the email is only stored, never delivered. Sent mail counts as read.
    row = Email(sender=MY_ADDRESS, recipient=email.recipient, subject=email.subject,
                body=email.body, timestamp=datetime.now(), read=True)
    db.add(row)
    db.commit()
    return row


@app.get("/emails", response_model=list[EmailOut])
def list_emails(db: Session = Depends(get_db)):
    return db.scalars(_newest_first(select(Email))).all()


# Fixed routes must come before /emails/{email_id}, or FastAPI would treat "unread" as an id
@app.get("/emails/unread", response_model=list[EmailOut])
def list_unread(db: Session = Depends(get_db)):
    return db.scalars(_newest_first(select(Email).where(Email.read.is_(False)))).all()


@app.get("/emails/search", response_model=list[EmailOut])
def search(q: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    pattern = f"%{q}%"
    condition = or_(Email.subject.ilike(pattern), Email.body.ilike(pattern), Email.sender.ilike(pattern))
    return db.scalars(_newest_first(select(Email).where(condition))).all()


@app.get("/emails/filter", response_model=list[EmailOut])
def filter_emails(
    recipient: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    db: Session = Depends(get_db),
):
    query = select(Email)
    if recipient:
        query = query.where(Email.recipient.ilike(recipient))
    if date_from:
        query = query.where(Email.timestamp >= datetime.combine(date_from, datetime.min.time()))
    if date_to:  # inclusive: everything before the start of the next day
        query = query.where(Email.timestamp < datetime.combine(date_to + timedelta(days=1), datetime.min.time()))
    return db.scalars(_newest_first(query)).all()


@app.get("/emails/{email_id}", response_model=EmailOut)
def get_email(email_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, email_id)


@app.patch("/emails/{email_id}/read", response_model=EmailOut)
def mark_read(email_id: int, db: Session = Depends(get_db)):
    email = _get_or_404(db, email_id)
    email.read = True
    db.commit()
    return email


@app.patch("/emails/{email_id}/unread", response_model=EmailOut)
def mark_unread(email_id: int, db: Session = Depends(get_db)):
    email = _get_or_404(db, email_id)
    email.read = False
    db.commit()
    return email


@app.delete("/emails/{email_id}")
def delete_email(email_id: int, db: Session = Depends(get_db)):
    email = _get_or_404(db, email_id)
    deleted = {"id": email.id, "subject": email.subject, "sender": email.sender}
    db.delete(email)
    db.commit()
    return {"deleted": True, **deleted}


@app.get("/reset_database")
def reset_database():
    return {"status": "reset", "emails": reset_db()}


# --------------------------------------------------------------------------------------
# Running the service
# --------------------------------------------------------------------------------------
def is_running() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=1).json().get("service") == SERVICE_NAME
    except (requests.RequestException, ValueError):
        return False


def ensure_running(timeout: float = 15.0) -> str:
    """Return the service URL, starting the service in a background thread if needed."""
    if is_running():
        return BASE_URL

    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="email-service")
    thread.start()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running():
            return BASE_URL
        if not thread.is_alive():
            break
        time.sleep(0.1)
    raise RuntimeError(
        f"Could not start the email service on {BASE_URL}. "
        f"Is port {PORT} used by another program? Set EMAIL_API_PORT to a free port."
    )


if __name__ == "__main__":
    print(f"Simulated email service on http://{HOST}:{PORT}  (API docs: http://{HOST}:{PORT}/docs)")
    uvicorn.run(app, host=HOST, port=PORT)
