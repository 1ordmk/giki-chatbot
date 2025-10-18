# 🚀 GIKI Chatbot - Deployment Guide

Complete guide to deploy your chatbot to production using free-tier services.

---

## 📋 Table of Contents

1. [Backend Deployment (AWS EC2)](#backend-aws-ec2)
2. [Frontend Deployment (Vercel/Netlify)](#frontend-deployment)
3. [Alternative: Render.com (All-in-One)](#alternative-rendercom)
4. [Database Setup](#database-setup)
5. [Domain & SSL](#domain--ssl)
6. [Monitoring & Maintenance](#monitoring)

---

## 🖥️ Backend Deployment (AWS EC2)

### Step 1: Launch EC2 Instance (Free Tier)

1. **Go to AWS Console** → EC2 → Launch Instance

2. **Configure Instance:**
   - Name: `giki-chatbot-backend`
   - AMI: Ubuntu Server 22.04 LTS (Free tier eligible)
   - Instance type: `t2.micro` (1 GB RAM, Free tier)
   - Key pair: Create new or use existing
   - Security Group: Allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS), 5000 (API)

3. **Launch** and wait for instance to start

### Step 2: Connect to Instance

```bash
# Download your key pair (.pem file)
chmod 400 your-key.pem

# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### Step 3: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and tools
sudo apt install python3-pip python3-venv git nginx -y

# Install Chrome and ChromeDriver (for scraping)
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb -y
```

### Step 4: Clone and Setup Project

```bash
# Clone your repository
git clone https://github.com/your-username/giki-chatbot.git
cd giki-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install production server
pip install gunicorn
```

### Step 5: Configure Environment

```bash
# Create .env file
nano .env

# Add your configuration:
PINECONE_API_KEY=your-pinecone-key
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-key
SECRET_KEY=your-random-secret-key
```

### Step 6: Process Data

```bash
# Generate and process data
python complete_project_setup.py
python preprocessing.py
```

### Step 7: Setup Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/giki-chatbot.service
```

Add this content:

```ini
[Unit]
Description=GIKI Chatbot Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/giki-chatbot
Environment="PATH=/home/ubuntu/giki-chatbot/venv/bin"
ExecStart=/home/ubuntu/giki-chatbot/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 backend:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable giki-chatbot
sudo systemctl start giki-chatbot

# Check status
sudo systemctl status giki-chatbot
```

### Step 8: Configure Nginx (Reverse Proxy)

```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/giki-chatbot
```

Add this content:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # or use EC2 public IP

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/giki-chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 9: Test Backend

```bash
# Test from EC2
curl http://localhost:5000/api/health

# Test from outside
curl http://your-ec2-public-ip/api/health
```

---

## 🌐 Frontend Deployment

### Option 1: Vercel (Recommended)

1. **Prepare frontend:**
```bash
# Update API URL in index.html
# Change: const API_BASE_URL = 'http://localhost:5000/api';
# To: const API_BASE_URL = 'http://your-ec2-ip/api';
```

2. **Deploy:**
   - Go to https://vercel.com
   - Sign up with GitHub
   - Click "New Project"
   - Upload `index.html` or connect GitHub repo
   - Deploy!

3. **Custom Domain (Optional):**
   - Go to Project Settings → Domains
   - Add your domain
   - Update DNS records as shown

### Option 2: Netlify

1. **Deploy:**
   - Go to https://netlify.com
   - Drag and drop `index.html`
   - Or connect GitHub repo

2. **Configure:**
   - Update API_BASE_URL in index.html
   - Redeploy

### Option 3: AWS S3 + CloudFront

```bash
# Create S3 bucket
aws s3 mb s3://giki-chatbot-frontend

# Enable static website hosting
aws s3 website s3://giki-chatbot-frontend --index-document index.html

# Upload files
aws s3 sync . s3://giki-chatbot-frontend --exclude "*.py" --exclude "venv/*"

# Make public
aws s3 put-bucket-policy --bucket giki-chatbot-frontend --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::giki-chatbot-frontend/*"
  }]
}'
```

---

## 🎯 Alternative: Render.com (Easiest)

Deploy both backend and frontend on Render (Free tier available)

### Backend on Render:

1. **Create Web Service:**
   - Go to https://render.com
   - New → Web Service
   - Connect GitHub repo
   - Select branch

2. **Configure:**
   ```
   Name: giki-chatbot-backend
   Environment: Python 3
   Build Command: pip install -r requirements.txt && python -m spacy download en_core_web_sm
   Start Command: gunicorn -w 4 -b 0.0.0.0:$PORT backend:app
   ```

3. **Add Environment Variables:**
   - PINECONE_API_KEY
   - LLM_PROVIDER
   - GROQ_API_KEY
   - SECRET_KEY

4. **Deploy** and get your backend URL

### Frontend on Render:

1. **Create Static Site:**
   - New → Static Site
   - Connect GitHub repo

2. **Configure:**
   ```
   Name: giki-chatbot-frontend
   Build Command: (leave empty)
   Publish Directory: .
   ```

3. **Update index.html:**
   - Replace API_BASE_URL with your Render backend URL

---

## 💾 Database Setup

### Pinecone (Vector Database)

Already configured in preprocessing.py - no additional setup needed!

### Student Data (SQLite → PostgreSQL)

For production, use PostgreSQL instead of JSON files:

```bash
# On EC2, install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Create database
sudo -u postgres psql
CREATE DATABASE giki_chatbot;
CREATE USER giki_admin WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE giki_chatbot TO giki_admin;
\q
```

Update `backend.py` to use PostgreSQL:

```python
import psycopg2

# Replace JSON loading with PostgreSQL
conn = psycopg2.connect(
    dbname="giki_chatbot",
    user="giki_admin",
    password="your-password",
    host="localhost"
)
```

---

## 🔒 Domain & SSL

### Get Free Domain:

- **Freenom**: https://www.freenom.com (free .tk, .ml, .ga domains)
- **No-IP**: https://www.noip.com (free subdomain)

### Setup SSL (Free with Let's Encrypt):

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (automatically configured)
sudo certbot renew --dry-run
```

Your site will now be accessible via https://your-domain.com

---

## 📊 Monitoring & Maintenance

### Setup Logging

```python
# In backend.py, add:
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
```

### Monitor System Resources

```bash
# Install monitoring tools
sudo apt install htop -y

# Check logs
sudo journalctl -u giki-chatbot -f

# Check Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Automated Backups

```bash
# Create backup script
nano backup.sh
```

```bash
#!/bin/bash
# Backup script for GIKI Chatbot

DATE=$(date +%Y%m%d)
BACKUP_DIR="/home/ubuntu/backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup data
tar -czf $BACKUP_DIR/giki-data-$DATE.tar.gz /home/ubuntu/giki-chatbot/data

# Backup database (if using PostgreSQL)
# pg_dump giki_chatbot > $BACKUP_DIR/db-$DATE.sql

# Keep only last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

```bash
# Make executable
chmod +x backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add: 0 2 * * * /home/ubuntu/giki-chatbot/backup.sh
```

### Setup Monitoring Dashboard (Optional)

```bash
# Install Prometheus and Grafana for advanced monitoring
# Or use simple monitoring:

# Create health check script
nano health_check.sh
```

```bash