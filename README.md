# SchemeScout 

Multilingual AI agent that matches citizens to government schemes using LangGraph and Gemini.

## Run locally
```
pip install -r requirements.txt
cp .env.example .env   # add your keys
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs

## Example
```
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Main 45 saal ka kisan hoon, UP mein 2 acre zameen hai, salana aay 1 lakh"}'
```

## Deploy (Render)
Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
Add GEMINI_API_KEY and TAVILY_API_KEY under Environment.

Results are indicative only; confirm on official portals.
