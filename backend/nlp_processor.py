import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self, data_path: str = "data/dataset.csv", chroma_db_path: str = "chroma_db", max_rows: int = 500):
        """
        Inicializa o processador NLP com ChromaDB
        """
        self.chroma_db_path = chroma_db_path
        self.data_path = data_path
        self.max_rows = max_rows
        
        # Modelo multilíngue (Português + Inglês)
        self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Cliente ChromaDB
        self.client = PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Pipeline de sentimento (multilíngue)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        # Pipeline de classificação zero-shot (multilíngue)
        self.classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli"
        )
        
        self.df = None
        self.collection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Carrega dados e inicializa ChromaDB"""
        try:
            # Carregar dataset
            self.df = pd.read_csv(self.data_path)
            self.df = self._normalize_dataframe(self.df)
            logger.info(f"Dataset carregado: {len(self.df)} textos")
            
            # Criar/obter coleção
            self.collection = self.client.get_or_create_collection(
                name="textos_dataset",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Verificar se já existem embeddings
            if self.collection.count() == 0:
                logger.info("Gerando embeddings para o dataset...")
                self._add_embeddings_to_chroma()
            else:
                logger.info(f"Embeddings já existem: {self.collection.count()} textos")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar database: {e}")
            raise

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza dataset para colunas padrão: texto, categoria, idioma"""
        columns_lower_map = {col.lower(): col for col in df.columns}

        text_candidates = ["texto", "text", "prompt", "content", "sentence", "message", "review"]
        text_col = next((columns_lower_map[c] for c in text_candidates if c in columns_lower_map), None)

        if text_col is None:
            object_cols = [col for col in df.columns if df[col].dtype == "object"]
            text_col = object_cols[0] if object_cols else None

        if text_col is None:
            raise ValueError("Nenhuma coluna de texto encontrada no dataset")

        if "categoria" in columns_lower_map:
            category_col = columns_lower_map["categoria"]
            categoria_series = df[category_col].astype(str)
        elif "inbound" in columns_lower_map:
            inbound_col = columns_lower_map["inbound"]
            categoria_series = np.where(df[inbound_col].astype(bool), "Inbound", "Outbound")
        else:
            categoria_series = "Geral"

        if "idioma" in columns_lower_map:
            language_col = columns_lower_map["idioma"]
            idioma_series = df[language_col].fillna("en").astype(str)
        elif "language" in columns_lower_map:
            language_col = columns_lower_map["language"]
            idioma_series = df[language_col].fillna("en").astype(str)
        else:
            idioma_series = "en"

        normalized_df = pd.DataFrame({
            "texto": df[text_col].astype(str),
            "categoria": categoria_series,
            "idioma": idioma_series,
        })

        normalized_df = normalized_df.dropna(subset=["texto"])
        normalized_df = normalized_df[normalized_df["texto"].str.strip() != ""]

        if len(normalized_df) > self.max_rows:
            logger.info(
                f"Dataset muito grande ({len(normalized_df)}). Limitando para {self.max_rows} linhas para inicialização rápida."
            )
            normalized_df = normalized_df.head(self.max_rows)

        return normalized_df.reset_index(drop=True)
    
    def _add_embeddings_to_chroma(self):
        """Adiciona embeddings ao ChromaDB"""
        textos = self.df['texto'].tolist()
        ids = [f"doc_{i}" for i in range(len(textos))]
        
        # Gerar embeddings em batch
        embeddings = self.embedding_model.encode(textos, show_progress_bar=True)
        
        # Adicionar ao ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=textos,
            metadatas=[
                {
                    "categoria": row['categoria'],
                    "idioma": row.get('idioma', 'pt')
                }
                for _, row in self.df.iterrows()
            ]
        )
        logger.info(f"✅ {len(textos)} textos adicionados ao ChromaDB")
    
    def buscar_textos_similares(self, prompt: str, top_k: int = 5) -> list:
        """
        Busca textos similares usando embeddings
        """
        try:
            # Gerar embedding do prompt
            prompt_embedding = self.embedding_model.encode([prompt])
            
            # Buscar no ChromaDB
            results = self.collection.query(
                query_embeddings=prompt_embedding.tolist(),
                n_results=top_k
            )
            
            # Processar resultados
            textos_similares = []
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distancia = results['distances'][0][i]
                similitude = 1 - (distancia / 2)  # Converter para score 0-1
                
                idx = int(doc_id.split('_')[1])
                
                textos_similares.append({
                    "id": doc_id,
                    "texto": results['documents'][0][i],
                    "categoria": results['metadatas'][0][i]['categoria'],
                    "similitude": float(similitude),
                    "idioma": results['metadatas'][0][i].get('idioma', 'pt')
                })
            
            return textos_similares
        
        except Exception as e:
            logger.error(f"Erro ao buscar textos similares: {e}")
            return []
    
    def classificar_texto(self, texto: str, categorias: list = None) -> dict:
        """
        Classifica o texto em uma das categorias disponíveis
        """
        try:
            if categorias is None:
                categorias = self.df['categoria'].unique().tolist()
            
            resultado = self.classifier(texto, categorias, multi_class=False)
            
            return {
                "categoria": resultado['labels'][0],
                "confianca": float(resultado['scores'][0]),
                "todas_categorias": [
                    {
                        "categoria": label,
                        "score": float(score)
                    }
                    for label, score in zip(resultado['labels'], resultado['scores'])
                ]
            }
        except Exception as e:
            logger.error(f"Erro ao classificar texto: {e}")
            return {"erro": str(e)}
    
    def analisar_sentimento(self, texto: str) -> dict:
        """
        Analisa o sentimento do texto (Positivo/Negativo/Neutro)
        """
        try:
            resultado = self.sentiment_pipeline(texto)
            
            label_map = {
                "1 star": "Muito Negativo",
                "2 stars": "Negativo",
                "3 stars": "Neutro",
                "4 stars": "Positivo",
                "5 stars": "Muito Positivo"
            }
            
            label = resultado[0]['label']
            sentimento = label_map.get(label, label)
            
            return {
                "sentimento": sentimento,
                "label_original": label,
                "confianca": float(resultado[0]['score'])
            }
        except Exception as e:
            logger.error(f"Erro ao analisar sentimento: {e}")
            return {"erro": str(e)}
    
    def analise_completa(self, prompt: str, top_k: int = 5) -> dict:
        """
        Realiza análise completa: similaridade + classificação + sentimento
        """
        return {
            "prompt": prompt,
            "textos_similares": self.buscar_textos_similares(prompt, top_k),
            "classificacao": self.classificar_texto(prompt),
            "sentimento": self.analisar_sentimento(prompt)
        }
