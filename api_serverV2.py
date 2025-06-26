from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import nest_asyncio
import uvicorn
import json
import traceback

nest_asyncio.apply()

# === Setup ===
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "scb10x/llama3.1-typhoon2-8b-instruct"
PERSIST_DIR = "chroma_db"

embedder = OllamaEmbeddings(model=EMBED_MODEL)
vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
llm = OllamaLLM(model=LLM_MODEL)

system_prompt = (
    "You are a financial analysis and stock expert. "
    "Respond ONLY in JSON format EXACTLY like this: "
    '{{"answer": "A or B or C or D or E or Rise or Fall", "raw_output": "<your full response>"}}. '
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

app = FastAPI()

class EvalRequest(BaseModel):
    question: str


@app.post("/eval")
async def evaluate(request: EvalRequest):
    try:
        docs = vector_db.similarity_search(request.question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        messages = prompt.format_messages(
            context=context,
            question=request.question
        )
        response = llm.invoke(messages)
        result = response.strip()

        # Try parsing response as JSON
        try:
            parsed = json.loads(result)
            answer = parsed.get("answer", "N/A")
            raw_output = parsed.get("raw_output", result)
        except json.JSONDecodeError:
            # Fallback: extract answer from plain text
            answer = "N/A"
            for option in ["A", "B", "C", "D", "E", "Rise", "Fall"]:
                if option in result:
                    answer = option
                    break
            raw_output = result

        return JSONResponse(content={
            "answer": answer,
            "raw_output": raw_output
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
