# BrainFlow Security Configuration

## Environment Variables

### Required for Production

| Variable | Description | Example |
|----------|-------------|---------|
| `NEUROLINKED_API_TOKEN` | Master API token for authentication | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `NEUROLINKED_REQUIRE_AUTH` | Require authentication for all endpoints | `true` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `NEUROLINKED_URL` | Brain server URL | `http://localhost:8000` |
| `NEUROLINKED_HOST` | Bind address | `127.0.0.1` |
| `NEUROLINKED_PORT` | Server port | `8000` |
| `NEUROLINKED_CORS_ORIGINS` | Allowed CORS origins | `http://localhost:8000` |
| `NEUROLINKED_RATE_LIMIT` | Enable rate limiting | `true` |
| `NEUROLINKED_RATE_LIMIT_REQUESTS` | Requests per window | `100` |
| `NEUROLINKED_RATE_LIMIT_WINDOW` | Window in seconds | `60` |
| `NEUROLINKED_ENV` | Environment mode | `development` |

## Quick Setup

### 1. Generate a Secure Token

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Set Environment Variables

**Linux/macOS:**
```bash
export NEUROLINKED_API_TOKEN="your-generated-token-here"
export NEUROLINKED_REQUIRE_AUTH="true"
export NEUROLINKED_ENV="production"
```

**Windows:**
```powershell
$env:NEUROLINKED_API_TOKEN="your-generated-token-here"
$env:NEUROLINKED_REQUIRE_AUTH="true"
$env:NEUROLINKED_ENV="production"
```

### 3. Configure MCP Server

Add the token to your MCP server configuration:

```json
{
  "mcpServers": {
    "brainflow": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "NEUROLINKED_API_TOKEN": "your-generated-token-here",
        "NEUROLINKED_URL": "http://localhost:8000"
      }
    }
  }
}
```

## Security Headers

The server automatically adds these security headers:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Content-Security-Policy: default-src 'self'; ...`
- `Strict-Transport-Security: max-age=31536000` (production only)

## Rate Limiting

By default, the server allows 100 requests per 60 seconds per IP address.
Configure with:

```bash
export NEUROLINKED_RATE_LIMIT_REQUESTS=100
export NEUROLINKED_RATE_LIMIT_WINDOW=60
```

## CORS Configuration

For production, restrict CORS to your domain:

```bash
export NEUROLINKED_CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
```

## Token Security

- Tokens are compared using constant-time comparison to prevent timing attacks
- Never commit tokens to version control
- Use environment variables or secure secret management

## P1: Advanced Configuration

### Knowledge Store Retention

Prevent unbounded growth of the knowledge store:

```bash
# Maximum entries to keep (oldest auto-pruned)
export NEUROLINKED_KNOWLEDGE_MAX_ENTRIES="10000"

# Delete entries older than N days
export NEUROLINKED_KNOWLEDGE_RETENTION_DAYS="90"
```

### Screen Observer Privacy

Exclude sensitive windows from screen capture:

```bash
# Comma-separated list of window title keywords to exclude
export NEUROLINKED_SCREEN_EXCLUDED_TITLES="password,pass,secret,login,credential,bitwarden,1password,lastpass,dashlane"
```

PII is automatically redacted from OCR text:
- Credit card numbers
- Social Security numbers
- Email addresses
- Passwords, secrets, tokens, keys

### Distributed Rate Limiting

For multi-worker deployments, use Redis:

```bash
export NEUROLINKED_REDIS_URL="redis://localhost:6379/0"
```

### Activity Log Retention

```bash
export NEUROLINKED_ACTIVITY_LOG_MAX="1000"
```

### Prometheus Metrics

Enable the `/metrics` endpoint:

```bash
export NEUROLINKED_METRICS_ENABLED="true"
export NEUROLINKED_METRICS_PORT="9090"
```

Metrics exposed:
- `brainflow_steps_total` - Total brain steps
- `brainflow_steps_per_second` - Current simulation rate
- `brainflow_neuromodulator_level` - Neuromodulator levels
- `brainflow_connected_clients` - WebSocket client count
- `brainflow_simulation_running` - Simulation state
- Rotate tokens periodically
- Use different tokens for different environments

## Production Checklist

- [ ] `NEUROLINKED_API_TOKEN` is set to a secure random value
- [ ] `NEUROLINKED_REQUIRE_AUTH=true`
- [ ] `NEUROLINKED_ENV=production`
- [ ] CORS origins are restricted to your domain
- [ ] Rate limiting is enabled
- [ ] Server binds to localhost or uses reverse proxy
- [ ] HTTPS is used (via reverse proxy)
- [ ] Tokens are not in code or logs
