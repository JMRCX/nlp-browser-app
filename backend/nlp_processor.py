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
    def __init__(self, data_path: str = "data/dataset.csv", chroma_db_path: str = "chroma_db"):
        """
        Inicializa o processador NLP com ChromaDB
        """
        self.chroma_db_path = chroma_db_path
        self.data_path = data_path
        
        # Modelo multilíngue (Português + Inglês)
        self.embedding_model = SentenceTransformer("sentence-transformers/multilingual-MiniLM-L12-v2")
        
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
