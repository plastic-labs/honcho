"""
Admin Dashboard Backend for KlimaShift
Provides APIs for system monitoring, agent creation, and cost analytics
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import httpx
import asyncpg
from datetime import datetime, timedelta
import yaml

app = FastAPI(title="KlimaShift Admin Dashboard API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
HONCHO_API_URL = os.getenv("HONCHO_API_URL", "http://api:8000")
HERMES_API_URL = os.getenv("HERMES_API_URL", "http://klimashift-agent-creator:8080")
DB_CONNECTION_URI = os.getenv("DB_CONNECTION_URI", "postgresql://postgres:postgres@database:5432/postgres")
AGENT_TEMPLATES_DIR = os.getenv("AGENT_TEMPLATES_DIR", "/app/agent-templates")

# Database connection pool
db_pool = None


async def get_db_pool():
    """Get database connection pool"""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            dsn=DB_CONNECTION_URI.replace("postgresql+psycopg://", "postgresql://"),
            min_size=2,
            max_size=10
        )
    return db_pool


# === Models ===

class SystemStats(BaseModel):
    total_peers: int
    total_sessions: int
    total_messages: int
    messages_last_24h: int
    queue_pending: int
    queue_processing: int
    queue_completed: int


class CostAnalytics(BaseModel):
    total_tokens_30d: int
    estimated_cost_30d: float
    cost_by_provider: Dict[str, float]
    daily_usage: List[Dict[str, Any]]
    top_peers: List[Dict[str, Any]]


class AgentTemplate(BaseModel):
    name: str
    description: str
    file_path: str
    parameters: List[str]


class AgentDeployRequest(BaseModel):
    template_name: str
    agent_name: str
    parameters: Dict[str, Any]


# === Routes ===

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard HTML"""
    return FileResponse("/app/dashboard.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# === System Monitoring ===

@app.get("/api/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # Get counts
        total_peers = await conn.fetchval("SELECT COUNT(*) FROM peers")
        total_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")
        
        # Messages in last 24 hours
        messages_24h = await conn.fetchval(
            "SELECT COUNT(*) FROM messages WHERE created_at > NOW() - INTERVAL '24 hours'"
        )
        
        # Queue stats
        queue_completed = await conn.fetchval(
            "SELECT COUNT(*) FROM queue WHERE processed = true"
        ) or 0
        
        queue_pending = await conn.fetchval(
            "SELECT COUNT(*) FROM queue WHERE processed = false"
        ) or 0
        
        return SystemStats(
            total_peers=total_peers or 0,
            total_sessions=total_sessions or 0,
            total_messages=total_messages or 0,
            messages_last_24h=messages_24h or 0,
            queue_pending=queue_pending,
            queue_processing=0,  # Not tracked in Honcho
            queue_completed=queue_completed
        )


@app.get("/api/services/status")
async def get_services_status():
    """Check status of all services"""
    services = {}
    
    # Check Honcho API
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{HONCHO_API_URL}/health")
            services["honcho_api"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": HONCHO_API_URL
            }
    except Exception as e:
        services["honcho_api"] = {"status": "error", "error": str(e)}
    
    # Check Hermes Gateway
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{HERMES_API_URL}/health")
            services["hermes_gateway"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": HERMES_API_URL
            }
    except Exception as e:
        services["hermes_gateway"] = {"status": "error", "error": str(e)}
    
    # Check Database
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            services["database"] = {"status": "healthy"}
    except Exception as e:
        services["database"] = {"status": "error", "error": str(e)}
    
    return services


@app.get("/api/queue/recent")
async def get_recent_queue_items(limit: int = 20):
    """Get recent queue items"""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        items = await conn.fetch(
            """
            SELECT id, processed, task_type, created_at
            FROM queue
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit
        )
        
        return [
            {
                "id": item['id'],
                "status": "completed" if item['processed'] else "pending",
                "task_type": item['task_type'],
                "created_at": item['created_at'].isoformat(),
                "updated_at": item['created_at'].isoformat()  # Honcho doesn't track updated_at
            }
            for item in items
        ]


# === Cost Analytics ===

@app.get("/api/analytics/cost", response_model=CostAnalytics)
async def get_cost_analytics():
    """Get cost analytics for last 30 days"""
    pool = await get_db_pool()
    
    # Cost per 1K tokens
    COSTS = {
        "gemini-flash": 0.000075,
        "claude-haiku": 0.0008,
        "claude-sonnet": 0.003,
        "openai-embedding": 0.00002
    }
    
    async with pool.acquire() as conn:
        # Total tokens last 30 days
        total_tokens = await conn.fetchval(
            """
            SELECT COALESCE(SUM(token_count), 0)
            FROM messages
            WHERE created_at > NOW() - INTERVAL '30 days'
            """
        ) or 0
        
        # Estimate cost (40% deriver at Gemini, 60% dialectic at Claude Haiku)
        deriver_tokens = int(total_tokens * 0.4)
        dialectic_tokens = int(total_tokens * 0.6)
        
        deriver_cost = (deriver_tokens / 1000) * COSTS["gemini-flash"]
        dialectic_cost = (dialectic_tokens / 1000) * COSTS["claude-haiku"]
        embedding_cost = (total_tokens / 1000) * COSTS["openai-embedding"]
        
        total_cost = deriver_cost + dialectic_cost + embedding_cost
        
        cost_by_provider = {
            "gemini": round(deriver_cost, 2),
            "claude": round(dialectic_cost, 2),
            "openai": round(embedding_cost, 2)
        }
        
        # Daily usage
        daily_data = await conn.fetch(
            """
            SELECT
                DATE(created_at) as date,
                COUNT(*) as messages,
                COALESCE(SUM(token_count), 0) as tokens
            FROM messages
            WHERE created_at > NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """
        )
        
        daily_usage = [
            {
                "date": row['date'].isoformat(),
                "messages": row['messages'],
                "tokens": row['tokens']
            }
            for row in daily_data
        ]
        
        # Top peers by usage
        top_peers_data = await conn.fetch(
            """
            SELECT
                p.name,
                COUNT(m.id) as message_count,
                COALESCE(SUM(m.token_count), 0) as total_tokens
            FROM messages m
            JOIN peers p ON m.peer_id = p.id
            WHERE m.created_at > NOW() - INTERVAL '30 days'
            GROUP BY p.id, p.name
            ORDER BY total_tokens DESC
            LIMIT 10
            """
        )
        
        top_peers = [
            {
                "name": row['name'],
                "messages": row['message_count'],
                "tokens": row['total_tokens']
            }
            for row in top_peers_data
        ]
        
        return CostAnalytics(
            total_tokens_30d=total_tokens,
            estimated_cost_30d=round(total_cost, 2),
            cost_by_provider=cost_by_provider,
            daily_usage=daily_usage,
            top_peers=top_peers
        )


# === Memory Management ===

@app.get("/api/peers")
async def get_peers(limit: int = 50):
    """Get list of peers"""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        peers = await conn.fetch(
            """
            SELECT
                p.id,
                p.name,
                p.created_at,
                COUNT(DISTINCT s.id) as session_count,
                COUNT(m.id) as message_count
            FROM peers p
            LEFT JOIN sessions s ON s.peer_id = p.id
            LEFT JOIN messages m ON m.peer_id = p.id
            GROUP BY p.id, p.name, p.created_at
            ORDER BY p.created_at DESC
            LIMIT $1
            """,
            limit
        )
        
        return [dict(peer) for peer in peers]


@app.get("/api/peers/{peer_id}/observations")
async def get_peer_observations(peer_id: str):
    """Get observations for a peer"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{HONCHO_API_URL}/v1/peers/{peer_id}/observations")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/peers/{peer_id}/dream")
async def trigger_dreamer(peer_id: str):
    """Trigger dreamer for a peer"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{HONCHO_API_URL}/v1/peers/{peer_id}/dream")
            response.raise_for_status()
            return {"status": "success", "message": "Dreamer triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Agent Creator ===

@app.get("/api/agent-templates", response_model=List[AgentTemplate])
async def list_agent_templates():
    """List available agent templates"""
    templates = []
    
    try:
        template_files = os.listdir(AGENT_TEMPLATES_DIR)
        
        for file in template_files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                file_path = os.path.join(AGENT_TEMPLATES_DIR, file)
                
                try:
                    with open(file_path, 'r') as f:
                        content = yaml.safe_load(f)
                        
                        templates.append(AgentTemplate(
                            name=file.replace('.yaml', '').replace('.yml', ''),
                            description=content.get('description', 'No description'),
                            file_path=file,
                            parameters=list(content.get('parameters', {}).keys())
                        ))
                except Exception as e:
                    print(f"Error loading template {file}: {e}")
                    
    except Exception as e:
        print(f"Error listing templates: {e}")
    
    return templates


@app.get("/api/agent-templates/{template_name}")
async def get_agent_template(template_name: str):
    """Get specific agent template content"""
    file_path = os.path.join(AGENT_TEMPLATES_DIR, f"{template_name}.yaml")
    
    if not os.path.exists(file_path):
        file_path = os.path.join(AGENT_TEMPLATES_DIR, f"{template_name}.yml")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/deploy")
async def deploy_agent(request: AgentDeployRequest):
    """Deploy a new agent from template"""
    # Load template
    template_path = os.path.join(AGENT_TEMPLATES_DIR, f"{request.template_name}.yaml")
    
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        with open(template_path, 'r') as f:
            template = yaml.safe_load(f)
        
        # Replace parameters in template
        agent_config = {
            "name": request.agent_name,
            "template": request.template_name,
            "config": template,
            "parameters": request.parameters
        }
        
        # For now, just return the config
        # TODO: Actually deploy to Hermes when API is ready
        return {
            "status": "success",
            "message": f"Agent '{request.agent_name}' configuration created",
            "agent_config": agent_config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Startup/Shutdown ===

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    await get_db_pool()
    print("✓ Database pool initialized")
    print(f"✓ Admin Dashboard Backend running")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global db_pool
    if db_pool:
        await db_pool.close()
        print("✓ Database pool closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
