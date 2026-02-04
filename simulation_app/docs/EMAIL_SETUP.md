# Email Setup Guide (Free)

This guide explains how to set up email delivery for the Behavioral Experiment Simulation Tool using free SMTP services.

## Quick Setup with Gmail (Recommended)

Gmail is the easiest option and is completely free.

### Step 1: Enable 2-Step Verification

1. Go to your [Google Account](https://myaccount.google.com/)
2. Click **Security** in the left sidebar
3. Under "Signing in to Google", click **2-Step Verification**
4. Follow the prompts to enable it (if not already enabled)

### Step 2: Create an App Password

1. Go to your [Google Account](https://myaccount.google.com/)
2. Click **Security** in the left sidebar
3. Under "Signing in to Google", click **App passwords**
   - If you don't see this option, make sure 2-Step Verification is enabled
4. Select **Mail** as the app
5. Select **Other** as the device and name it "Simulation Tool"
6. Click **Generate**
7. **Copy the 16-character password** (you'll only see it once!)

### Step 3: Configure Streamlit Secrets

Add the following to your `.streamlit/secrets.toml` file:

```toml
# Email Configuration (Gmail)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your.email@gmail.com"
SMTP_PASSWORD = "xxxx xxxx xxxx xxxx"  # Your 16-character app password
SMTP_FROM_EMAIL = "your.email@gmail.com"
SMTP_FROM_NAME = "Behavioral Experiment Simulation Tool"

# Where to send reports (Dr. Dimant's email)
INSTRUCTOR_NOTIFICATION_EMAIL = "edimant@sas.upenn.edu"
```

**Important:** Replace `your.email@gmail.com` with your actual Gmail address and `xxxx xxxx xxxx xxxx` with your app password.

### Step 4: Deploy to Streamlit Cloud

If deploying to Streamlit Cloud:
1. Go to your app settings
2. Click on **Secrets**
3. Paste the configuration from Step 3
4. Save and redeploy

---

## Alternative: University Email (UPenn Example)

If you prefer to use your university email:

```toml
# UPenn Email (example - check with IT for exact settings)
SMTP_SERVER = "smtp.office365.com"
SMTP_PORT = 587
SMTP_USERNAME = "your-pennkey@upenn.edu"
SMTP_PASSWORD = "your-password"
SMTP_FROM_EMAIL = "your-pennkey@upenn.edu"
INSTRUCTOR_NOTIFICATION_EMAIL = "edimant@sas.upenn.edu"
```

**Note:** Some universities require VPN or have specific authentication requirements. Check with your IT department.

---

## Alternative: Outlook/Hotmail

```toml
SMTP_SERVER = "smtp-mail.outlook.com"
SMTP_PORT = 587
SMTP_USERNAME = "your.email@outlook.com"
SMTP_PASSWORD = "your-password"
SMTP_FROM_EMAIL = "your.email@outlook.com"
INSTRUCTOR_NOTIFICATION_EMAIL = "edimant@sas.upenn.edu"
```

---

## Troubleshooting

### "SMTP authentication failed"
- **Gmail:** Make sure you're using an App Password, not your regular password
- **Other providers:** Check if 2FA is enabled and if an app password is required

### "Could not connect to SMTP server"
- Check that the SMTP_SERVER and SMTP_PORT are correct
- Some networks block SMTP ports - try a different network
- University networks may require VPN

### "Email not received"
- Check your spam/junk folder
- Verify the recipient email address is correct
- Some email providers have sending limits (Gmail: 500/day)

---

## Security Notes

- **Never share your app password** - it grants access to send email from your account
- **Use a dedicated email account** if you're concerned about security
- **App passwords can be revoked** anytime from your Google Account settings
- The password is stored securely in Streamlit Secrets (encrypted at rest)

---

## Why SMTP instead of SendGrid?

- **Free forever** - No paid tiers or sign-up required
- **No external service dependencies** - Uses Python's built-in libraries
- **Works immediately** - No account verification or domain configuration
- **Uses your existing email** - Gmail, Outlook, university email all work
