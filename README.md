# FarmWise AI

i wanna made an app that useful for farmers as AI Farmer Query Support System so i got the codes i just want you to integrate all in the one combined output in a perfect manner 1. Environment Configuration (Sanitized)
python
# backend/app/core/config.py
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "AI Farmer Query Support System"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security - MUST BE SET IN ENVIRONMENT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")  # ⚠️ REQUIRED
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    
    # Database - MUST BE SET IN ENVIRONMENT
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")  # ⚠️ REQUIRED
    
    # Redis - MUST BE SET IN ENVIRONMENT
    REDIS_URL: str = os.getenv("REDIS_URL", "")  # ⚠️ REQUIRED
    
    # AI Services - ALL FROM ENVIRONMENT
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Vector Database
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    
    # Weather APIs - ALL FROM ENVIRONMENT
    OPENWEATHER_API_KEY: Optional[str] = os.getenv("OPENWEATHER_API_KEY")
    IMD_API_KEY: Optional[str] = os.getenv("IMD_API_KEY")
    
    # Google Services - ALL FROM ENVIRONMENT
    GOOGLE_MAPS_API_KEY: Optional[str] = os.getenv("GOOGLE_MAPS_API_KEY")
    GOOGLE_CLOUD_API_KEY: Optional[str] = os.getenv("GOOGLE_CLOUD_API_KEY")
    
    # Storage - ALL FROM ENVIRONMENT
    CLOUDINARY_CLOUD_NAME: Optional[str] = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: Optional[str] = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: Optional[str] = os.getenv("CLOUDINARY_API_SECRET")
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
    
    # Firebase - ALL FROM ENVIRONMENT
    FIREBASE_API_KEY: Optional[str] = os.getenv("FIREBASE_API_KEY")
    FIREBASE_AUTH_DOMAIN: Optional[str] = os.getenv("FIREBASE_AUTH_DOMAIN")
    FIREBASE_PROJECT_ID: Optional[str] = os.getenv("FIREBASE_PROJECT_ID")
    FIREBASE_STORAGE_BUCKET: Optional[str] = os.getenv("FIREBASE_STORAGE_BUCKET")
    FIREBASE_MESSAGING_SENDER_ID: Optional[str] = os.getenv("FIREBASE_MESSAGING_SENDER_ID")
    FIREBASE_APP_ID: Optional[str] = os.getenv("FIREBASE_APP_ID")
    
    # Model Settings
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Knowledge Base
    KNOWLEDGE_BASE_PATH: str = os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_PERIOD: int = int(os.getenv("RATE_LIMIT_PERIOD", "60"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    def validate_required(self):
        """Validate required settings for production"""
        if self.ENVIRONMENT == "production":
            required = [
                "SECRET_KEY",
                "DATABASE_URL",
                "REDIS_URL"
            ]
            missing = [r for r in required if not getattr(self, r)]
            if missing:
                raise ValueError(f"Missing required settings: {', '.join(missing)}")
        
        # Warn about missing API keys but don't fail
        if not self.OPENAI_API_KEY and not self.GEMINI_API_KEY:
            print("⚠️ WARNING: No AI API keys configured. AI features will not work.")

settings = Settings()

# Validate on import if in production
if settings.ENVIRONMENT == "production":
    settings.validate_required()
2. AI Service (No Hardcoded Keys)
python
# backend/app/services/ai_service.py
import openai
import google.generativeai as genai
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from app.core.config import settings
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        """Initialize AI services using environment variables"""
        self.openai_client = None
        self.gemini_client = None
        self.rag_service = RAGService()
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AI clients from environment variables"""
        try:
            # Initialize OpenAI from env
            if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "":
                openai.api_key = settings.OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not configured")
            
            # Initialize Gemini from env
            if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "":
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_client = genai
                logger.info("Gemini client initialized successfully")
            else:
                logger.warning("Gemini API key not configured")
                
        except Exception as e:
            logger.error(f"Error initializing AI clients: {str(e)}")
    
    async def process_query(
        self,
        query: str,
        user_context: Dict[str, Any],
        language: str = "en",
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Process farmer query with full context"""
        try:
            # Check if any AI client is available
            if not self.openai_client and not self.gemini_client:
                return {
                    "response": "I apologize, but the AI service is currently unavailable. Please try again later.",
                    "error": "No AI clients available - Please configure API keys",
                    "confidence": 0.0
                }
            
            # 1. Retrieve relevant context from RAG
            rag_context = await self.rag_service.retrieve_context(
                query=query,
                user_context=user_context,
                top_k=5
            )
            
            # 2. Build prompts
            system_prompt = self._build_system_prompt(user_context, language)
            user_prompt = self._build_user_prompt(query, rag_context, user_context)
            
            # 3. Get response from LLM (with fallback)
            response = await self._get_llm_response(system_prompt, user_prompt, language)
            
            # 4. Parse response
            parsed_response = self._parse_response(response)
            
            # 5. Calculate confidence
            confidence = self._calculate_confidence(parsed_response, rag_context)
            
            return {
                "response": parsed_response.get("answer", ""),
                "sources": rag_context if include_sources else None,
                "confidence": confidence,
                "recommendations": parsed_response.get("recommendations", []),
                "disclaimer": "This is AI-generated advice. Please consult local agricultural experts for critical decisions.",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error. Please try again or rephrase your question.",
                "error": str(e) if settings.DEBUG else None,
                "confidence": 0.0
            }
    
    async def _get_llm_response(self, system_prompt: str, user_prompt: str, language: str) -> str:
        """Get response from LLM with fallback mechanism"""
        
        errors = []
        
        # Try OpenAI first (if available)
        if self.openai_client:
            try:
                response = await self.openai_client.ChatCompletion.acreate(
                    model=settings.DEFAULT_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            except Exception as e:
                errors.append(f"OpenAI: {str(e)}")
                logger.warning(f"OpenAI failed: {str(e)}")
        
        # Fallback to Gemini (if available)
        if self.gemini_client:
            try:
                model = self.gemini_client.GenerativeModel('gemini-pro')
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                errors.append(f"Gemini: {str(e)}")
                logger.warning(f"Gemini failed: {str(e)}")
        
        # If all fail
        error_msg = "All LLM providers failed: " + " | ".join(errors)
        raise Exception(error_msg)
3. Environment Template (NO KEYS)
bash
# backend/.env.example
# ⚠️ COPY THIS FILE TO .env AND FILL IN YOUR VALUES
# ⚠️ NEVER COMMIT .env TO VERSION CONTROL

# ============================================
# APPLICATION
# ============================================
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=CHANGE_THIS_TO_A_RANDOM_STRING
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# ============================================
# DATABASE
# ============================================
DATABASE_URL=postgresql://user:password@localhost:5432/farmer_db

# ============================================
# REDIS CACHE
# ============================================
REDIS_URL=redis://localhost:6379

# ============================================
# AI SERVICES - GET FROM YOUR PROVIDERS
# ============================================
# OpenAI: https://platform.openai.com/api-keys
OPENAI_API_KEY=

# Google Gemini: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=

# Anthropic Claude: https://console.anthropic.com/
ANTHROPIC_API_KEY=

# ============================================
# VECTOR DATABASE - QDRANT
# ============================================
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# ============================================
# WEATHER APIS
# ============================================
# OpenWeather: https://openweathermap.org/api
OPENWEATHER_API_KEY=

# IMD: Register with Indian Meteorological Department
IMD_API_KEY=

# ============================================
# GOOGLE SERVICES
# ============================================
# Google Maps: https://developers.google.com/maps
GOOGLE_MAPS_API_KEY=

# Google Cloud: https://console.cloud.google.com/
GOOGLE_CLOUD_API_KEY=

# ============================================
# CLOUD STORAGE
# ============================================
# Cloudinary: https://cloudinary.com/
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=

# Supabase: https://supabase.com/
SUPABASE_URL=
SUPABASE_KEY=

# ============================================
# FIREBASE AUTH
# ============================================
FIREBASE_API_KEY=
FIREBASE_AUTH_DOMAIN=
FIREBASE_PROJECT_ID=
FIREBASE_STORAGE_BUCKET=
FIREBASE_MESSAGING_SENDER_ID=
FIREBASE_APP_ID=

# ============================================
# MODEL SETTINGS
# ============================================
DEFAULT_LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small

# ============================================
# KNOWLEDGE BASE
# ============================================
KNOWLEDGE_BASE_PATH=./data/knowledge_base

# ============================================
# RATE LIMITING
# ============================================
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
4. Frontend Environment Template (NO KEYS)
bash
# frontend/.env.local.example
# ⚠️ COPY THIS FILE TO .env.local AND FILL IN YOUR VALUES

# ============================================
# API ENDPOINTS
# ============================================
NEXT_PUBLIC_API_URL=http://localhost:8000/api
NEXT_PUBLIC_WEBSOCKET_URL=ws://localhost:8000/ws

# ============================================
# GOOGLE SERVICES
# ============================================
NEXT_PUBLIC_GOOGLE_MAPS_KEY=
NEXT_PUBLIC_GOOGLE_ANALYTICS_ID=

# ============================================
# CLOUDINARY
# ============================================
NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME=

# ============================================
# SUPABASE
# ============================================
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=

# ============================================
# FEATURE FLAGS
# ============================================
NEXT_PUBLIC_ENABLE_VOICE=true
NEXT_PUBLIC_ENABLE_OFFLINE=true
NEXT_PUBLIC_ENABLE_PWA=true
5. Docker Compose with Environment Variables
yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-production}
      - DEBUG=${DEBUG:-false}
      - DATABASE_URL=postgresql://${DB_USER:-user}:${DB_PASSWORD:-password}@postgres:5432/${DB_NAME:-farmer_db}
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
      # API Keys passed from environment
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
      - GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY}
      - CLOUDINARY_API_KEY=${CLOUDINARY_API_KEY}
      - CLOUDINARY_API_SECRET=${CLOUDINARY_API_SECRET}
      - SECRET_KEY=${SECRET_KEY}
    env_file:
      - backend/.env
    depends_on:
      - postgres
      - redis
      - qdrant
    networks:
      - app-network

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000/api
      - NEXT_PUBLIC_GOOGLE_MAPS_KEY=${GOOGLE_MAPS_API_KEY}
    depends_on:
      - backend
    networks:
      - app-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=${DB_USER:-user}
      - POSTGRES_PASSWORD=${DB_PASSWORD:-password}
      - POSTGRES_DB=${DB_NAME:-farmer_db}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - app-network

  qdrant:
    image: qdrant/qdrant:v1.5.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:
  qdrant_data:

networks:
  app-network:
    driver: bridge
6. Docker Compose Override for Production (No Keys)
yaml
# docker-compose.prod.yml
version: '3.8'

services:
  backend:
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
    # Keys are passed from GitHub Secrets or environment

  frontend:
    environment:
      - NEXT_PUBLIC_API_URL=https://api.yourdomain.com
7. GitHub Actions with Secrets (NO KEYS IN CODE)
yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build Backend
      uses: docker/build-push-action@v4
      with:
        context: ./backend
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/farmer-backend:latest
        # Pass secrets as build args (they won't be in the image)
        build-args: |
          OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}
          GEMINI_API_KEY=${{ secrets.GEMINI_API_KEY }}
          SECRET_KEY=${{ secrets.SECRET_KEY }}
    
    - name: Build Frontend
      uses: docker/build-push-action@v4
      with:
        context: ./frontend
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/farmer-frontend:latest
    
    - name: Deploy
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USER }}
        key: ${{ secrets.SERVER_SSH_KEY }}
        script: |
          cd /app/ai-farmer-query-system
          # Pull latest code
          git pull
          # Create .env from secrets
          echo "OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}" > backend/.env
          echo "GEMINI_API_KEY=${{ secrets.GEMINI_API_KEY }}" >> backend/.env
          echo "SECRET_KEY=${{ secrets.SECRET_KEY }}" >> backend/.env
          echo "DATABASE_URL=${{ secrets.DATABASE_URL }}" >> backend/.env
          echo "REDIS_URL=redis://redis:6379" >> backend/.env
          echo "OPENWEATHER_API_KEY=${{ secrets.OPENWEATHER_API_KEY }}" >> backend/.env
          echo "GOOGLE_MAPS_API_KEY=${{ secrets.GOOGLE_MAPS_API_KEY }}" >> backend/.env
          echo "CLOUDINARY_API_KEY=${{ secrets.CLOUDINARY_API_KEY }}" >> backend/.env
          echo "CLOUDINARY_API_SECRET=${{ secrets.CLOUDINARY_API_SECRET }}" >> backend/.env
          echo "ENVIRONMENT=production" >> backend/.env
          echo "DEBUG=false" >> backend/.env
          # Deploy
          docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
          docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
          docker system prune -f
8. .gitignore (Prevent Key Leakage)
gitignore
# .gitignore

# Environment files with secrets
.env
.env.local
.env.production
.env.*.local

# Backend
backend/.env
backend/*.env
backend/*.key
backend/*.pem

# Frontend
frontend/.env
frontend/.env.local
frontend/.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Node modules
node_modules/
.pnpm-store/

# Next.js
.next/
out/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# Docker
*.pid
*.sock

# Database
*.db
*.sqlite
*.sqlite3

# Secrets
*.pem
*.key
*.crt
*.p12
*.pfx
secrets/
keys/
9. Kubernetes Secrets (Production Deployment)
yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: farmer-app-secrets
type: Opaque
# These will be encrypted in the cluster
# Values are base64 encoded in actual deployment
stringData:
  secret-key: "YOUR_SECRET_KEY"
  openai-api-key: "YOUR_OPENAI_KEY"
  gemini-api-key: "YOUR_GEMINI_KEY"
  database-url: "postgresql://user:pass@postgres:5432/db"
  redis-url: "redis://redis:6379"
  openweather-api-key: "YOUR_WEATHER_KEY"
  google-maps-key: "YOUR_MAPS_KEY"
  cloudinary-api-key: "YOUR_CLOUDINARY_KEY"
  cloudinary-api-secret: "YOUR_CLOUDINARY_SECRET"
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: farmer-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: farmer-backend
  template:
    metadata:
      labels:
        app: farmer-backend
    spec:
      containers:
      - name: backend
        image: your-registry/farmer-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: farmer-app-secrets
              key: secret-key
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: farmer-app-secrets
              key: openai-api-key
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: farmer-app-secrets
              key: gemini-api-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: farmer-app-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: farmer-app-secrets
              key: redis-url
        - name: OPENWEATHER_API_KEY
          valueFrom:
            secretKeyRef:
              name: farmer-app-secrets
              key: openweather-api-key
        - name: CLOUDINARY_API_KEY
          valueFrom:
            secretKeyRef:
              name: farmer-app-secrets
              key: cloudinary-api-key
        - name: CLOUDINARY_API_SECRET
          valueFrom:
            secretKeyRef:
              name: farmer-app-secrets
              key: cloudinary-api-secret
        - name: ENVIRONMENT
          value: "production"
        - name: DEBUG
          value: "false"
10. Setup Instructions (No Keys in Code)
Step 1: Clone Repository
bash
git clone https://github.com/yourusername/ai-farmer-query-system.git
cd ai-farmer-query-system
Step 2: Create Environment Files
bash
# Backend
cp backend/.env.example backend/.env
# Edit backend/.env with your actual values

# Frontend
cp frontend/.env.local.example frontend/.env.local
# Edit frontend/.env.local with your actual values
Step 3: Add Your API Keys
bash
# Edit backend/.env and add your keys:
# OPENAI_API_KEY=your_actual_key_here
# GEMINI_API_KEY=your_actual_key_here
# SECRET_KEY=your_random_secret_string
# DATABASE_URL=your_database_connection_string
# REDIS_URL=your_redis_connection_string
Step 4: For Lovable Cloud Deployment
Go to Project Settings → Secrets

Add each key as a secret:

OPENAI_API_KEY

GEMINI_API_KEY

SECRET_KEY

DATABASE_URL

REDIS_URL

OPENWEATHER_API_KEY

GOOGLE_MAPS_API_KEY

CLOUDINARY_API_KEY

CLOUDINARY_API_SECRET

The secrets will be injected as environment variables

Step 5: Deploy
bash
# Local development
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
✅ Key Security Practices (Summary)
NEVER hardcode API keys in source code

NEVER commit .env files to version control

ALWAYS use environment variables or secrets managers

ALWAYS use different keys for dev/production

REGULARLY rotate API keys

MONITOR API key usage for anomalies

USE rate limiting to prevent abuse

ENCRYPT sensitive data at rest and in transit

📋 API Keys You'll Need to Configure
Service	Where to Get
OpenAI	https://platform.openai.com/api-keys
Gemini	https://makersuite.google.com/app/apikey
OpenWeather	https://openweathermap.org/api
Google Maps	https://developers.google.com/maps
Cloudinary	https://cloudinary.com/
Supabase	https://supabase.com/
Firebase	https://console.firebase.google.com/

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://farmmingwiseai.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/029fb3a1-aec8-4d65-9c97-b460152b0f88).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
