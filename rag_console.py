#!/usr/bin/env python3
# rag_console.py — búsqueda semántica sobre datos musicales

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ------------------ CONFIG ------------------
DATA_FILE = "music_data.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3  # número de resultados más similares que se mostrarán
# --------------------------------------------

# ---------- CARGAR DATOS ----------
def load_music_data():
    path = Path(DATA_FILE)
    if not path.exists():
        print("⚠️ No se encuentra el archivo music_data.json. Ejecuta primero scrape_music_rag.py.")
        exit()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------- CARGAR MODELO ----------
def load_model():
    print("🧠 Cargando modelo de embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    return model


# ---------- BUSCAR LOS MÁS SIMILARES ----------
def semantic_search(query, docs, model, top_k=TOP_K):
    # Vectoriza la pregunta
    query_emb = model.encode([query], convert_to_numpy=True)[0]

    # Extrae todos los embeddings de los documentos
    doc_embs = np.array([d["embedding"] for d in docs])

    # Calcula similitud coseno
    similarities = np.dot(doc_embs, query_emb) / (np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb))

    # Ordena por similitud (mayor primero)
    top_indices = similarities.argsort()[::-1][:top_k]

    # Devuelve los documentos más parecidos
    return [(docs[i], float(similarities[i])) for i in top_indices]


# ---------- MAIN ----------
def main():
    print("🎧 Bienvenido al buscador musical RAG (por consola)")
    docs = load_music_data()
    model = load_model()

    while True:
        query = input("\n🔍 Escribe tu pregunta (o 'salir' para terminar): ").strip()
        if query.lower() == "salir":
            print("👋 Adiós!")
            break

        results = semantic_search(query, docs, model)
        print("\n🎶 Resultados más parecidos:")
        for doc, score in results:
            print(f"\n🎵 {doc['title']} — {doc['artist']}")
            print(f"   🔗 {doc['url']}")
            meta = doc.get("metadata", {})
            if meta:
                print(f"   📀 {meta.get('genre', 'Género desconocido')} | {meta.get('country', '')} | {meta.get('released', '')}")
            print(f"   🔢 Similitud: {score:.3f}")


if __name__ == "__main__":
    main()
