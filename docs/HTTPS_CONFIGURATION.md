# HTTPS Configuration Guide for ASTINA

This guide provides instructions for configuring HTTPS/TLS for secure deployment of the ASTINA application.

## Overview

HTTPS (Hypertext Transfer Protocol Secure) is essential for:
- Encrypting data in transit
- Preventing man-in-the-middle attacks
- Protecting user credentials and sensitive health data
- Meeting compliance requirements (HIPAA, UU PDP)

## Deployment Options

### Option 1: Cloud Run with HTTPS (Recommended)

Google Cloud Run automatically provides HTTPS termination.

#### Configuration
```yaml
# cloudbuild.yaml or gcloud command
gcloud run deploy astina \
  --image gcr.io/PROJECT_ID/astina:latest \
  --platform managed \
  --region REGION \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 1 \
  --timeout 3600 \
  --allow-unauthenticated
```

Cloud Run automatically:
- Provides SSL/TLS certificates
- Handles HTTPS termination
- Renews certificates automatically
- Enforces HTTPS (redirects HTTP to HTTPS)

### Option 2: Streamlit with Nginx Reverse Proxy

For self-hosted deployments, use Nginx as a reverse proxy with SSL.

#### Prerequisites
- Nginx installed
- SSL certificate (Let's Encrypt recommended)
- Domain name configured

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/astina
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # HSTS (HTTP Strict Transport Security)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### Enable Configuration
```bash
sudo ln -s /etc/nginx/sites-available/astina /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Option 3: Streamlit Built-in SSL (Development Only)

For local development with SSL:

```python
# .streamlit/config.toml
[server]
sslCertFile = "path/to/cert.pem"
sslKeyFile = "path/to/key.pem"
enableXsrfProtection = true
enableCORS = false
```

#### Generate Self-Signed Certificate (Development)
```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

## SSL Certificate Management

### Let's Encrypt (Free, Recommended)

#### Installation
```bash
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx
```

#### Obtain Certificate
```bash
sudo certbot --nginx -d your-domain.com
```

#### Auto-Renewal
```bash
sudo certbot renew --dry-run
# Certbot automatically sets up cron job for renewal
```

### Commercial SSL Certificates

For production, consider commercial SSL providers:
- DigiCert
- GlobalSign
- Comodo

## Security Headers

### HSTS (HTTP Strict Transport Security)
```nginx
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

### Content Security Policy
```nginx
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;
```

### X-Frame-Options
```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
```

### X-Content-Type-Options
```nginx
add_header X-Content-Type-Options "nosniff" always;
```

### Referrer-Policy
```nginx
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

## Streamlit Configuration for HTTPS

### Environment Variables
```bash
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
```

### config.toml
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501
```

## Verification

### Test HTTPS Configuration
```bash
# Check SSL certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Check HTTP to HTTPS redirect
curl -I http://your-domain.com

# Check security headers
curl -I https://your-domain.com
```

### SSL Labs Test
Visit https://www.ssllabs.com/ssltest/ and test your domain for:
- Certificate validity
- Protocol support
- Cipher strength
- Overall rating (aim for A+)

## Compliance Requirements

### HIPAA
- Encryption in transit (TLS 1.2+)
- Unique certificates
- Certificate rotation
- No deprecated protocols (SSLv2, SSLv3, TLS 1.0, TLS 1.1)

### UU PDP No. 27/2022
- Data encryption in transit
- Secure communication channels
- Certificate management
- Security measures documentation

## Troubleshooting

### Certificate Not Trusted
- Ensure certificate chain is complete
- Check certificate expiration
- Verify intermediate certificates

### Mixed Content Errors
- Ensure all resources use HTTPS
- Check for hardcoded HTTP URLs
- Verify proxy configuration

### WebSocket Connection Issues
```nginx
# Ensure WebSocket support in Nginx
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
```

## Security Best Practices

1. **Always use HTTPS in production**
2. **Use TLS 1.2 or higher** (disable TLS 1.0/1.1)
3. **Use strong ciphers** (AES-256, ChaCha20)
4. **Enable HSTS** with preload
5. **Implement certificate monitoring**
6. **Automate certificate renewal**
7. **Use certificate pinning** for critical APIs
8. **Regular security audits** (SSL Labs, security headers)

## Monitoring

### Certificate Expiry Monitoring
```bash
# Check certificate expiry
echo | openssl s_client -connect your-domain.com:443 2>/dev/null | openssl x509 -noout -dates
```

### Security Headers Monitoring
```bash
# Check security headers
curl -I https://your-domain.com | grep -E "Strict-Transport-Security|Content-Security-Policy|X-Frame-Options"
```

## References

- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [OWASP TLS Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy/)

---

**Last Updated:** 2026-10-25  
**Document Version:** 1.0
