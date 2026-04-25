# NeuroLinked Brain - Claude Integration

This project has a neuromorphic brain running at http://localhost:8000.

## Authentication

All API endpoints require Bearer token authentication. Set the `NEUROLINKED_API_TOKEN` environment variable:

```bash
export NEUROLINKED_API_TOKEN="your-token-here"
```

Include the token in requests:
```bash
curl -H "Authorization: Bearer $NEUROLINKED_API_TOKEN" http://localhost:8000/api/claude/summary
```

## Quick API Reference

All endpoints require `Authorization: Bearer <token>` header.

- GET /api/health - Health check (no auth required)
- GET /api/version - Version info (no auth required)
- GET /api/claude/summary - Read brain state
- POST /api/claude/observe - Send observations (body: {"type":"text","content":"...","source":"claude"})
- GET /api/claude/insights - Get brain insights
- POST /api/brain/save - Save brain state
