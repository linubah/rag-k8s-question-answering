from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import nltk
import sys
import os
import pickle
import pandas as pd
from collections import defaultdict

PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJ_DIR)
from data.data import generate_dataset


class Retriever:
    def __init__(self, embeddings_file='doc_embeddings.pkl'):
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        
        self.embeddings_file = embeddings_file

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../data/docs.csv")
        self.docs_data, self.docs = self.load_documents_from_csv(data_dir)

        tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in self.docs]

        self.model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.doc_embeddings = self._load_or_compute_embeddings()

    def _load_or_compute_embeddings(self):
        if os.path.exists(f"rag/{self.embeddings_file}"):
            with open(self.embeddings_file, 'rb') as f:
                return pickle.load(f)
        else:
            embeddings = self.model.encode(self.docs)
            
            with open(f"rag/{self.embeddings_file}", 'wb') as f:
                pickle.dump(embeddings, f)
            
            return embeddings

    def aggregate_document_scores(self, chunks, scores):
        doc_scores = defaultdict(float)
        for chunk, score in zip(chunks, scores):
            doc_scores[chunk["document_id"]] += score
        return doc_scores
    
    def truncate_to_token_limit(self, text, max_tokens):
        tokens = nltk.word_tokenize(text)
        truncated_tokens = tokens[:max_tokens]
        return " ".join(truncated_tokens)

    def create_context(self, doc_ids, max_total_tokens=7000, max_tokens_per_chunk=700):
        contexts = []
        resources = []
        total_tokens = 0

        for doc_id in doc_ids:
            related_chunks = [
                doc for doc in self.docs_data if doc["document_id"] == doc_id
            ]
            related_chunks = sorted(related_chunks, key=lambda x: x["order"])
            resources.append(related_chunks[0]["source"])
            context = ""
            for chunk in related_chunks:
                truncated_chunk = self.truncate_to_token_limit(chunk["content"], max_tokens_per_chunk)
                chunk_tokens = len(nltk.word_tokenize(truncated_chunk))

                if total_tokens + chunk_tokens <= max_total_tokens:
                    context += truncated_chunk + "\n\n"
                    total_tokens += chunk_tokens
                else:
                    break
            if context.strip():
                contexts.append(context.strip())
            
            if total_tokens >= max_total_tokens:
                break

        return contexts, resources



    def get_docs(self, query: str, num_chunks: int=12, top_docs: int=1):
        tokenized_query = nltk.word_tokenize(query.lower())

        bm25_scores = self.bm25.get_scores(tokenized_query)
        semantic_scores = self._get_semantic_similarity_scores(query)

        scores = [0.3 * bm25_score + 0.7 * semantic_score for bm25_score, semantic_score in zip(bm25_scores, semantic_scores)][0]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_chunks = [self.docs_data[i] for i in top_indices[:num_chunks]]

        doc_scores = self.aggregate_document_scores(top_chunks, [scores[i] for i in top_indices[:num_chunks]])
        sorted_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        selected_doc_ids = [doc_id for doc_id, _ in sorted_doc_ids[:top_docs]]
        result, resources = self.create_context(selected_doc_ids)

        return result, resources
    
    def _get_bm25_scores(self, query: str):
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        return scores
    
    def _get_semantic_similarity_scores(self, query: str):
        query_embedding = self.model.encode(query)
        scores = self.model.similarity(query_embedding, self.doc_embeddings)
        return scores
    
    def load_documents_from_csv(self, file_path: str) -> list[str]:
        if not os.path.exists(file_path):
            links_df = pd.read_csv("data/data_links.csv")
            links = links_df['link'].tolist()
            generate_dataset(links)

        df = pd.read_csv(file_path)
        docs = df.to_dict(orient='records')
        contents = df['content'].tolist()
        return docs, contents


        
    