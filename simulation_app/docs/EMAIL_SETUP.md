# Email Setup Guide (Free SMTP)

## Configuration for edimant@sas.upenn.edu

### Streamlit Secrets

```toml
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "edimant@sas.upenn.edu"
SMTP_PASSWORD = "your-16-char-app-password"
SMTP_FROM_EMAIL = "edimant@sas.upenn.edu"
INSTRUCTOR_NOTIFICATION_EMAIL = "edimant@sas.upenn.edu"
```

### Getting the App Password

1. Sign into [https://myaccount.google.com/](https://myaccount.google.com/) with `edimant@sas.upenn.edu`
2. Go to **Security** → **2-Step Verification** (enable if needed)
3. Go to **App passwords**: [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
4. Create: App = **Mail**, Device = **Other** → "Simulation Tool"
5. Copy the 16-character password (use as SMTP_PASSWORD)

---

## Troubleshooting

**"App passwords" not available?**
- University IT may have disabled this feature
- Contact UPenn IT to request access, or use a personal Gmail as sender

**Authentication failed?**
- Must use App Password, not regular password
- 2-Step Verification must be enabled first
- Password should be 16 characters (spaces optional)
