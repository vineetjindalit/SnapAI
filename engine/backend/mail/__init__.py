"""Email sending abstraction. SNAPPY_EMAIL_BACKEND=mailgun|sendgrid|console."""
from .sender import EmailSender, get_email_sender, EmailMessage  # noqa: F401
