from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CAMINHO_DB= 'db'

prompt_template = """
Responda a pergunta do usuário:
{pergunta}

com base nas seguintes informações:
{base_conhecimento}
"""

class perguntaRequest(BaseModel):
    pergunta: str


try:
    print("Inicializando banco de dados e modelos...")
    funcao_embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = Chroma(persist_directory=CAMINHO_DB,
    embedding_function=funcao_embedding)
    print("Sistema inicializado com sucesso.")
except Exception as e:
    print("Erro ao inicializar GoogleGenerativeAIEmbeddings:", e)
    db = None


@app.post("/api/chat")
async def chat_endpoint(request: perguntaRequest):
    if not db:
        raise HTTPException(status_code=500, detail="Banco de dados não está disponível.")

    pergunta = request.pergunta

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=4)  


    if len(resultados) == 0 or resultados[0][1] < 0.5:
        return {
            "resposta": "Desculpe, não tenho informações suficientes para responder a essa pergunta no momento.",
            "sucesso": False
        }

    texto_resultado = [res[0].page_content for res in resultados]
    base_conhecimento = "\n\n----\n\n".join(texto_resultado)


    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    resposta_ai = chain.invoke({
        "pergunta": pergunta,
        "base_conhecimento": base_conhecimento
    })

    return {
        "resposta": resposta_ai.content,
        "sucesso": True
    }

