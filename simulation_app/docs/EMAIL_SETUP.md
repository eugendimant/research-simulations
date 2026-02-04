# Email Setup Guide (Free)

## Recommended: Use a Personal Gmail to Send

Since university Google Workspace often has restrictions, the simplest solution is:
1. Create/use a **personal Gmail account** to SEND emails
2. Emails are delivered TO your university address (edimant@sas.upenn.edu)

### Step 1: Set Up a Personal Gmail

Use any personal Gmail account (e.g., `yourgmail@gmail.com`).

### Step 2: Enable 2-Step Verification

1. Go to [https://myaccount.google.com/security](https://myaccount.google.com/security)
2. Click **2-Step Verification** → Enable it

### Step 3: Create App Password

1. Go to [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
2. Select app: **Mail**
3. Select device: **Other** → "Simulation Tool"
4. Click **Generate**
5. Copy the 16-character password

### Step 4: Configure Streamlit Secrets

```toml
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "yourgmail@gmail.com"
SMTP_PASSWORD = "abcd efgh ijkl mnop"
SMTP_FROM_EMAIL = "yourgmail@gmail.com"
INSTRUCTOR_NOTIFICATION_EMAIL = "edimant@sas.upenn.edu"
```

**Result:** Emails are sent FROM your personal Gmail TO your university email.

---

## Troubleshooting

**"App passwords" not visible?**
- 2-Step Verification must be enabled first
- Direct link: [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)

**Still not working?**
- Make sure you're using the App Password (16 chars), not your regular Gmail password
- Check that SMTP_USERNAME matches the Gmail account that generated the App Password
