markdown
# 🧠 NLP Browser App

Uma aplicação web inteligente para análise de textos usando IA, com suporte para Português e Inglês.

## ✨ Funcionalidades

- 🔍 **Busca de Textos Similares**: Encontra textos semelhantes no dataset usando embeddings vetorizados
- 📂 **Classificação de Texto**: Classifica automaticamente textos em categorias predefinidas
- 😊 **Análise de Sentimento**: Detecta o sentimento do texto (Positivo/Negativo/Neutro)
- 🧠 **Modelos Multilíngues**: Suporte completo para Português e Inglês
- 📊 **Análise Completa**: Executa todas as análises simultaneamente

## 🛠️ Tecnologias

### Backend
- **FastAPI**: Framework web rápido e moderno
- **ChromaDB**: Banco de dados vetorial para embeddings
- **SentenceTransformers**: Modelos pré-treinados para embeddings multilíngues
- **HuggingFace Transformers**: Modelos de IA pré-treinados

### Frontend
- **HTML5**: Estrutura semântica
- **CSS3**: Design responsivo e moderno
- **JavaScript**: Interatividade e chamadas à API

## 📦 Instalação

### Pré-requisitos
- Python 3.11+
- Node.js (opcional, para development server)

### 1. Clone o repositório

```bash
git clone https://github.com/JMRCX/nlp-browser-app.git
cd nlp-browser-app
```

### 2. Setup Backend

```bash
# Criar ambiente virtual
py -V:Astral\CPython3.11.14 -m venv .venv

# Ativar ambiente virtual
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Instalar dependências
pip install -r backend/requirements.txt

# Opcional: acelerar indexação inicial de embeddings (default já é 500)
# Windows (PowerShell)
$env:NLP_MAX_ROWS="500"
# macOS/Linux
export NLP_MAX_ROWS=500
```

### 3. Executar o Backend

```bash
python backend/app.py
```

O backend estará disponível em `http://localhost:8000`

Swagger UI: `http://localhost:8000/docs`

### 4. Abrir o Frontend

```bash
# Em outra aba do terminal, abra o arquivo
cd ../frontend
```

Abra `index.html` no seu navegador ou use um servidor local:

```bash
# Python 3.6+
python -m http.server 8080

# Acesse em http://localhost:8080
```

## 🚀 Como Usar

1. **Digite um texto** na caixa de input
2. **Escolha uma análise**:
   - 📊 Análise Completa: Executa tudo de uma vez
   - 🔍 Textos Similares: Encontra textos parecidos no dataset
   - 📂 Classificar: Classifica em categorias
   - 😊 Sentimento: Analisa o sentimento
3. **Visualize os resultados** em tempo real

### Atalhos
- **Ctrl + Enter**: Executa análise completa

## 📊 API Endpoints

### POST `/buscar_similares`
Busca textos similares no dataset

**Request:**
```json
{
  "prompt": "Seu texto aqui",
  "top_k": 5
}
```

**Response:**
```json
{
  "sucesso": true,
  "prompt": "...",
  "quantidade": 5,
  "textos": [
    {
      "id": "doc_0",
      "texto": "...",
      "categoria": "Positivo",
      "similitude": 0.95,
      "idioma": "pt"
    }
  ]
}
```

### POST `/classificar`
Classifica um texto

**Request:**
```json
{
  "prompt": "Seu texto aqui"
}
```

**Response:**
```json
{
  "sucesso": true,
  "prompt": "...",
  "classificacao": {
    "categoria": "Positivo",
    "confianca": 0.92,
    "todas_categorias": [...]
  }
}
```

### POST `/sentimento`
Analisa sentimento

**Request:**
```json
{
  "prompt": "Seu texto aqui"
}
```

**Response:**
```json
{
  "sucesso": true,
  "prompt": "...",
  "sentimento": {
    "sentimento": "Positivo",
    "label_original": "5 stars",
    "confianca": 0.89
  }
}
```

### POST `/analise_completa`
Executa análise completa

**Response:**
```json
{
  "sucesso": true,
  "resultado": {
    "prompt": "...",
    "textos_similares": [...],
    "classificacao": {...},
    "sentimento": {...}
  }
}
```

## 🗄️ Dataset

O arquivo `backend/data/dataset.csv` é usado como fonte de textos para embeddings.

O backend normaliza automaticamente o dataset para o formato interno `texto`, `categoria`, `idioma`.
Se as colunas padrão não existirem, ele tenta mapear nomes comuns como:

- texto: `texto`, `text`, `prompt`, `content`, `sentence`, `message`, `review`
- categoria: `categoria` (ou usa `inbound` para gerar `Inbound`/`Outbound`)
- idioma: `idioma` ou `language` (fallback: `en`)

**Formato:**
```csv
texto,categoria,idioma
"Este é um ótimo produto.",Positivo,pt
"This product is amazing!",Positivo,en
"Não gostei.",Negativo,pt
```

### Adicionar seus próprios textos

1. Edite `backend/data/dataset.csv`
2. Preferencialmente use colunas `texto,categoria,idioma` (ou um dos nomes aceitos)
3. Delete a pasta `backend/chroma_db` para regenerar embeddings
4. Reinicie o backend

## 🎨 Customização

### Modelos de Embeddings
Em `nlp_processor.py`, line ~25:
```python
self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

Outras opções:
- `paraphrase-multilingual-MiniLM-L12-v2`
- `multilingual-e5-small`
- `multilingual-e5-base`

### Modelos de Sentimento
Em `nlp_processor.py`, line ~31:
```python
self.sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
```

### Categorias Padrão
O sistema detecta automaticamente as categorias do CSV. Para forçar categorias específicas, edite a função `classificar_texto()`.

## 📝 Estrutura de Diretórios

```
nlp-browser-app/
├── backend/
│   ├── app.py                 # FastAPI app principal
│   ├── nlp_processor.py       # Lógica NLP
│   ├── requirements.txt       # Dependências Python
│   ├── data/
│   │   └── dataset.csv        # Dados de exemplo
│   └── chroma_db/             # Vector store (gerado automaticamente)
├── frontend/
│   ├── index.html             # Interface web
│   ├── style.css              # Estilos
│   └── script.js              # Lógica frontend
├── .gitignore
└── README.md
```

## 🔧 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Erro: CORS quando abrir o frontend
Verifique se o backend está rodando em `http://localhost:8000`

### Embeddings lentando na primeira execução
Normal! O download dos modelos leva alguns minutos. Será cacheado depois.

### Backend não inicia com erro `_ssl` no Windows
Recrie o ambiente virtual com Python 3.11 e reinstale dependências:
```bash
Remove-Item -Recurse -Force .venv
py -V:Astral\CPython3.11.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/app.py
```

### Erro de memória com modelos grandes
Use modelos menores:
```python
SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
```

## 📚 Referências

- [ChromaDB Docs](https://docs.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [HuggingFace Models](https://huggingface.co/models)

## 📄 Licença

MIT License - Sinta-se livre para usar e modificar!

## 👤 Autor

Desenvolvido por **JMRCX** em 2026

## 🤝 Contribuições

Contribuições são bem-vindas! Faça um fork e crie um pull request.


# Ou abra index.html direto no navegador
```
