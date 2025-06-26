# ✅ 1. Import modules
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import nest_asyncio
import uvicorn

# ✅ 2. Patch event loop for Jupyter
nest_asyncio.apply()

# ✅ 3. Load your vector DB + model (same as your LLM pipeline)
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = "chroma_db"
LLM_MODEL = "scb10x/llama3.1-typhoon2-8b-instruct"

embedder = OllamaEmbeddings(model=EMBED_MODEL)
vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
llm = OllamaLLM(model=LLM_MODEL)
agent = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

# ✅ 4. Setup API app
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/eval")
async def evaluate(request: QuestionRequest):
    try:
        q = request.question.strip()
        result = agent.run(q).strip()

        # Simple logic to extract answer
        for option in ["A", "B", "C", "D", "Rise", "Fall"]:
            if option in result:
                answer = option
                break
        else:
            answer = "N/A"

        return JSONResponse(content={
            "answer": answer,
            "raw_output": result
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ✅ 5. Run the app (in notebook)
uvicorn.run(app, host="0.0.0.0", port=4000)