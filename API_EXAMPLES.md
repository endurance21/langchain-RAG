# RAG API - cURL Examples

## Base URL
```
http://localhost:8000
```

## 1. Root Endpoint (API Info)
```bash
curl -X GET http://localhost:8000/
```

## 2. Health Check
```bash
curl -X GET http://localhost:8000/health
```

## 3. Query RAG System (Basic)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key security considerations mentioned in the document?"
  }'
```

## 4. Query RAG System (With RAG disabled - just LLM)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "use_rag": false
  }'
```

## 5. Query RAG System (Custom document count)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key security considerations mentioned in the document?",
    "use_rag": true,
    "k": 10
  }'
```

## 6. Pretty Print JSON Response
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key security considerations mentioned in the document?"
  }' | python -m json.tool
```

## 7. Save Response to File
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key security considerations mentioned in the document?"
  }' -o response.json
```

## Example Responses

### Health Check Response
```json
{
  "status": "healthy",
  "rag_initialized": true
}
```

### Query Response
```json
{
  "answer": "Based on the document, the key security considerations include...",
  "query": "What are the key security considerations mentioned in the document?",
  "use_rag": true
}
```

