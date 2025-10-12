#!/usr/bin/env python3
# scrape_music_rag.py
# Scraper musical de Discogs sin login + embeddings

import time
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer

# ------------------ CONFIG ------------------
BASE_URL = "https://www.discogs.com"
SEARCH_QUERY = "rock"           # Puedes cambiarlo por jazz, pop, etc.
OUTPUT_FILE = "music_data.json"
MAX_PAGES = 3
EMBED_MODEL = "all-MiniLM-L6-v2"
HEADLESS = False                # True = sin abrir ventana; False = muestra navegador
SLOW_MO_MS = 200
# --------------------------------------------

def scrape_music_site(max_pages=MAX_PAGES):
    print("🎵 Iniciando scraping musical en Discogs...")
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, slow_mo=SLOW_MO_MS)
        page = browser.new_page()

        # --- PÁGINA DE BÚSQUEDA ---
        search_url = f"{BASE_URL}/search/?q={SEARCH_QUERY}&type=release"
        print(f"🔍 Navegando a: {search_url}")
        page.goto(search_url, timeout=60000)
        page.wait_for_selector(".card, .search_result, article", timeout=60000)

        # --- RECORRER PÁGINAS ---
        for page_idx in range(max_pages):
            print(f"📄 Procesando página {page_idx + 1}...")
            items = page.query_selector_all(".card, .search_result, article")
            print(f"   Encontrados {len(items)} elementos en la página.")

            for idx, it in enumerate(items):
                try:
                    title_el = it.query_selector("h4, .card__title, .search_result_title")
                    title = title_el.inner_text().strip() if title_el else ""

                    artist_el = it.query_selector(".artist, .card__artist, .card__subtitle, .search_result_artist")
                    artist = artist_el.inner_text().strip() if artist_el else ""

                    year_el = it.query_selector(".year, .release-year")
                    year = year_el.inner_text().strip() if year_el else ""

                    anchor = it.query_selector("a")
                    href = anchor.get_attribute("href") if anchor else ""
                    url = href if href.startswith("http") else (BASE_URL + href) if href else ""

                    meta = {}
                    label_el = it.query_selector(".label, .card__label")
                    if label_el:
                        meta["label"] = label_el.inner_text().strip()
                    genre_el = it.query_selector(".genre")
                    if genre_el:
                        meta["genre"] = genre_el.inner_text().strip()

                    text_blob = " | ".join(filter(None, [title, artist, year, meta.get("genre", ""), meta.get("label", "")]))
                    doc_id = f"pg{page_idx}_i{idx}_{int(time.time())}"

                    results.append({
                        "doc_id": doc_id,
                        "source": BASE_URL,
                        "title": title,
                        "artist": artist,
                        "year": year,
                        "url": url,
                        "metadata": meta,
                        "text": text_blob
                    })
                except Exception as e:
                    print(f"   ⚠️ Error parseando item {idx}: {e}")
                    continue

            # --- PAGINACIÓN ---
            try:
                next_btn = page.query_selector('a[rel="next"], a.pagination_next, .pagination-next, .next')
                if next_btn:
                    next_href = next_btn.get_attribute("href")
                    if next_href:
                        next_url = next_href if next_href.startswith("http") else (BASE_URL + next_href)
                        print(f"   → Siguiente página: {next_url}")
                        page.goto(next_url, timeout=60000)
                        page.wait_for_selector(".card, .search_result, article", timeout=60000)
                        page.wait_for_timeout(1000)
                    else:
                        print("   🚫 No hay más páginas.")
                        break
                else:
                    print("   🚫 No hay más páginas.")
                    break
            except Exception as e:
                print(f"   ⚠️ Error en paginación: {e}")
                break

        browser.close()

    print(f"✅ Scraping finalizado. Total: {len(results)} elementos.")
    return results


def embed_music_data(docs, model_name=EMBED_MODEL):
    print("🧠 Generando embeddings con", model_name)
    model = SentenceTransformer(model_name)
    texts = [d.get("text", "") for d in docs]
    if not texts:
        print("⚠️ No hay textos a indexar.")
        return docs
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for i, d in enumerate(docs):
        d["embedding"] = embeddings[i].tolist()
    print("✅ Embeddings completados.")
    return docs


def save_json(data, filename=OUTPUT_FILE):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Datos guardados en {filename}")


def main():
    docs = scrape_music_site(max_pages=MAX_PAGES)
    if not docs:
        print("⚠️ No se extrajo ningún documento. Revisa los selectores.")
        return
    embedded = embed_music_data(docs)
    save_json(embedded, OUTPUT_FILE)
    print("🎶 Pipeline completado.")


if __name__ == "__main__":
    main()
