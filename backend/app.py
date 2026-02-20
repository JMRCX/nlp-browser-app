from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nlp_processor import NLPProcessor
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
NLP_MAX_ROWS = int(os.getenv("NLP_MAX_ROWS", "500"))

# Inicializar FastAPI
app = FastAPI(title="NLP Browser App", version="1.0.0")

# CORS - Permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar NLP Processor
try:
    nlp = NLPProcessor(
        data_path=DATASET_PATH,
        chroma_db_path=CHROMA_DB_PATH,
        max_rows=NLP_MAX_ROWS
    )
    logger.info("✅ NLP Processor inicializado com sucesso")
except Exception as e:
    logger.error(f"❌ Erro ao inicializar NLP Processor: {e}")
    nlp = None

# Modelos Pydantic
class PromptRequest(BaseModel):
    prompt: str
    top_k: int = 5

class AnaliseSimplesRequest(BaseModel):
    prompt: str

# Rotas
@app.get("/")
def root():
    return {
        "mensagem": "NLP Browser App API",
        "versao": "1.0.0",
        "endpoints": [
            "/docs - Documentação Swagger",
            "/buscar_similares - Buscar textos similares",
            "/classificar - Classificar texto",
            "/sentimento - Analisar sentimento",
            "/analise_completa - Análise completa"
        ]
    }

@app.post("/buscar_similares")
def buscar_similares(request: PromptRequest):
    """Busca textos similares no dataset"""
    if not nlp:
        raise HTTPException(status_code=500, detail="NLP Processor não inicializado")
    
    try:
        resultados = nlp.buscar_textos_similares(request.prompt, request.top_k)
        return {
            "sucesso": True,
            "prompt": request.prompt,
            "quantidade": len(resultados),
            "textos": resultados
        }
    except Exception as e:
        logger.error(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classificar")
def classificar(request: AnaliseSimplesRequest):
    """Classifica o texto"""
    if not nlp:
        raise HTTPException(status_code=500, detail="NLP Processor não inicializado")
    
    try:
        resultado = nlp.classificar_texto(request.prompt)
        return {
            "sucesso": True,
            "prompt": request.prompt,
            "classificacao": resultado
        }
    except Exception as e:
        logger.error(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentimento")
def analisar_sentimento(request: AnaliseSimplesRequest):
    """Analisa o sentimento do texto"""
    if not nlp:
        raise HTTPException(status_code=500, detail="NLP Processor não inicializado")
    
    try:
        resultado = nlp.analisar_sentimento(request.prompt)
        return {
            "sucesso": True,
            "prompt": request.prompt,
            "sentimento": resultado
        }
    except Exception as e:
        logger.error(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analise_completa")
def analise_completa(request: PromptRequest):
    """Realiza análise completa"""
    if not nlp:
        raise HTTPException(status_code=500, detail="NLP Processor não inicializado")
    
    try:
        resultado = nlp.analise_completa(request.prompt, request.top_k)
        return {
            "sucesso": True,
            "resultado": resultado
        }
    except Exception as e:
        logger.error(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "online",
        "nlp_initialized": nlp is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
