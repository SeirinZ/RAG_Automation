"""
OpenAI-compatible API wrapper for Open WebUI integration
With conversation history support + streaming
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncIterator
import time
import logging
import json
import asyncio

from .retrieval import retriever
from .config import settings
from .models import model_manager

logger = logging.getLogger(__name__)

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

def build_context_aware_query(messages: List[Message]) -> str:
    """Build query with conversation context"""
    
    user_messages = [msg for msg in messages if msg.role == "user"]
    if not user_messages:
        return ""
    
    current_question = user_messages[-1].content
    
    # If first message, no context needed
    if len(messages) <= 1:
        return current_question
    
    # Build conversation context
    conversation = []
    for msg in messages[:-1]:  # All except current
        if msg.role == "user":
            conversation.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            # Brief context (first 150 chars)
            brief = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            conversation.append(f"Assistant: {brief}")
    
    # Limit context to last 5 exchanges (10 messages)
    if len(conversation) > 10:
        conversation = conversation[-10:]
    
    # Build enhanced query
    context_prompt = "\n".join(conversation)
    enhanced_query = f"""Previous conversation:
{context_prompt}

Current question: {current_question}

Based on the conversation above, answer the current question."""
    
    logger.info(f"  📝 Context from {len(conversation)} messages")
    return enhanced_query

async def stream_ollama_response(query: str, simple_query: str) -> AsyncIterator[str]:
    """
    Stream response from Ollama token by token

    Args:
        query: Full query (may include conversation context for LLM)
        simple_query: Simple query (just current question for retrieval)

    Yields OpenAI-compatible SSE chunks
    """

    chat_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    try:
        # Use FULL RAG pipeline for retrieval (same as /query endpoint)
        logger.info(f"  🔍 Retrieving for: {simple_query[:80]}...")

        # Use the proper retrieval pipeline with query expansion
        result = retriever.query(simple_query, use_reranking=True)
        docs = result.get('source_documents', [])

        # Build context from retrieved docs
        context = "\n\n".join([doc.page_content for doc in docs[:settings.TOP_K]])
        
        # Build prompt
        prompt_text = model_manager.prompt.format(context=context, question=query)
        
        # Stream from Ollama
        logger.info("  🌊 Starting stream...")
        
        # Ollama streaming
        from langchain_community.llms import Ollama
        streaming_llm = Ollama(
            model=settings.LLM_MODEL,
            temperature=0,
            callbacks=[]  # Can add streaming callbacks
        )
        
        # Buffer for accumulating response
        full_response = ""
        
        # Stream tokens
        for chunk in streaming_llm.stream(prompt_text):
            if chunk:
                full_response += chunk
                
                # Format as OpenAI SSE chunk
                chunk_data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.LLM_MODEL,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": chunk
                        },
                        "finish_reason": None
                    }]
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.01)
        
        # Send final chunk with finish_reason
        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": settings.LLM_MODEL,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
        logger.info(f"  ✓ Stream complete ({len(full_response)} chars)")
        
    except Exception as e:
        logger.error(f"❌ Stream error: {str(e)}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "stream_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion with streaming support"""
    
    try:
        if not request.messages:
            return {"error": "No messages provided"}
        
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return {"error": "No user message found"}
        
        # Build context-aware query
        query = build_context_aware_query(request.messages)
        
        current_q = user_messages[-1].content
        logger.info(f"🔍 Query: {current_q[:100]}...")
        logger.info(f"  📊 Messages: {len(request.messages)} | Stream: {request.stream}")
        
        # STREAMING RESPONSE
        if request.stream:
            return StreamingResponse(
                stream_ollama_response(query, simple_query=current_q),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        # NON-STREAMING RESPONSE (original behavior)
        else:
            result = retriever.query(query, use_reranking=True)
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=settings.LLM_MODEL,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result['result']
                    },
                    "finish_reason": "stop"
                }]
            )
            
            return response.dict()
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return {"error": str(e)}

@router.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)"""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.LLM_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }