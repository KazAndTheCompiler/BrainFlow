# NeuroLinked Brain - Claude Integration

This project has a neuromorphic brain running at http://localhost:8000.

## Quick API Reference
- GET /api/claude/summary - Read brain state
- POST /api/claude/observe - Send observations (body: {"type":"text","content":"...","source":"claude"})
- GET /api/claude/insights - Get brain insights
- POST /api/brain/save - Save brain state
