# Email Setup Guide (Free SMTP)

This guide explains how to configure email delivery using free SMTP.

## For Google Workspace (University Email like @sas.upenn.edu)

University email accounts that run through Google Workspace require an **App Password**.

### Step 1: Create an App Password

1. Go to [https://myaccount.google.com/](https://myaccount.google.com/) (sign in with your university account)
2. Click **Security** in the left sidebar
3. Under "Signing in to Google", find **2-Step Verification**
   - If not enabled, enable it first (required for App Passwords)
4. After 2-Step Verification is enabled, go back to Security
5. Click **App passwords** (may be under "2-Step Verification" section)
6. Select app: **Mail**
7. Select device: **Other** → name it "Simulation Tool"
8. Click **Generate**
9. **Copy the 16-character password** (spaces are optional)

**Note:** If you don't see "App passwords", your university IT may have disabled it. Contact them or use a personal Gmail instead.

### Step 2: Configure Streamlit Secrets

Add to `.streamlit/secrets.toml`:

```toml
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "edimant@sas.upenn.edu"
SMTP_PASSWORD = "xxxx xxxx xxxx xxxx"
SMTP_FROM_EMAIL = "edimant@sas.upenn.edu"
INSTRUCTOR_NOTIFICATION_EMAIL = "edimant@sas.upenn.edu"
```

Replace `xxxx xxxx xxxx xxxx` with your 16-character App Password.

---

## For Personal Gmail

Same process as above, but use your @gmail.com address.

---

## Troubleshooting

### "Authentication failed"
- You must use an **App Password**, not your regular password
- 2-Step Verification must be enabled first
- Check that SMTP_USERNAME matches the account that created the App Password

### "App passwords not available"
- Your organization may have disabled App Passwords
- Use a personal Gmail account instead
- Contact IT to request App Password access

### Email not received
- Check spam/junk folder
- Verify the recipient email is correct
- Gmail has a 500 email/day limit

---

## Configuration Reference

| Setting | Value |
|---------|-------|
| SMTP_SERVER | `smtp.gmail.com` |
| SMTP_PORT | `587` |
| SMTP_USERNAME | Your full email address |
| SMTP_PASSWORD | 16-character App Password |
| SMTP_FROM_EMAIL | Same as username |
