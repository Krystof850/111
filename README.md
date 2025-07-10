# Modular Speech-to-Text + Chat API

## Memory Optimized Version

This version uses OpenAI API for Whisper transcription instead of local models, dramatically reducing memory usage from 750MB to ~100MB.

## Features

- **OpenAI Whisper API**: Speech-to-text transcription
- **OpenAI GPT-4o-mini**: Chat responses
- **Memory optimized**: No local ML models
- **Modular architecture**: Isolated services
- **Railway deployment**: Production ready

## API Endpoints

- `GET /health` - Health check
- `POST /transcribe` - Speech-to-text
- `POST /chat` - AI chat
- `GET /services` - Service status

## Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key for Whisper and GPT
- `PORT` - Server port (default: 8000)

## Deployment

Optimized for Railway.app with automatic deployment from GitHub.# Force restart Thu Jul 10 12:55:17 PM UTC 2025
