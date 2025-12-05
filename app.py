"""
Rafiq-AI (version améliorée)
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
import re

import streamlit as st
import requests
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# try optional BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False

# -------------------- CONFIG --------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "rafiq_aiVF")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "knowledge_base")

DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.12))

# -------------------- CONNECT DB --------------------
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
knowledge_collection = mongo_db[MONGO_COLLECTION_NAME]

# -------------------- SESSION STATE INIT --------------------
if "knowledge_paragraphs" not in st.session_state:
    st.session_state.knowledge_paragraphs = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "model_warmed" not in st.session_state:
    st.session_state.model_warmed = False

if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD

if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = "BM25 (si dispo)" if BM25_AVAILABLE else "TF-IDF"

if "top_k" not in st.session_state:
    st.session_state.top_k = 3

# -------------------- HASSANIYA MINI DICTIONARY --------------------
HASSANIYA_DICTIONARY = {
    # Salutations
    "salam": "bonjour",
    "slm": "bonjour",
    "aslema": "bonjour",
    "ahlan": "bonjour",
    "marhaba": "bienvenue",
    "labass": "ça va",
    "lbass": "ça va",
    "kifak": "comment vas-tu",
    "kifhalek": "comment vas-tu",
    "kifhalou": "comment vas-tu",
    "bikhair": "bien",
    "hamdoullah": "grâce à Dieu",

    # Remerciements
    "shukran": "merci",
    "mersi": "merci",
    "barakallahoufik": "merci beaucoup",
    "yjazik": "que Dieu te récompense",

    # Accord
    "wakha": "d'accord",
    "waih": "oui",
    "eyh": "oui",
    "aywa": "oui",
    "ah": "oui",
    "la": "non",
    "mawah": "non",

    # Formules
    "inchallah": "si Dieu le veut",
    "inchaallah": "si Dieu le veut",
    "mashallah": "quelle belle chose",
    "bismillah": "au nom de Dieu",

    # Mots interrogatifs
    "waqtach": "quand",
    "waqtaš": "quand",
    "imta": "quand",
    "fin": "où",
    "wen": "où",
    "mnin": "d'où",
    "ach": "quoi",
    "chnou": "quoi",
    "shnou": "quoi",
    "alash": "pourquoi",
    "3lach": "pourquoi",
    "kayf": "comment",

    # Questions fréquentes pour ton IA
    "waqtach ybda": "quand commence",
    "waqtach ysali": "quand termine",
    "fin yskeun": "où se trouve",
    "ach howa": "qu'est-ce que",
    "chnou howa": "qu'est-ce que",

    # Temps
    "lyoum": "aujourd'hui",
    "baker": "demain",
    "ams": "hier",
    "daba": "maintenant",
    "f sabah": "ce matin",
    "f l3shia": "cet après-midi",
    "f lil": "cette nuit",

    # Besoin / demande
    "baghi": "je veux",
    "bghit": "je veux",
    "bgha": "il veut",
    "tbghini": "veux-tu",
    "a3tini": "donne-moi",
    "affak": "s'il te plaît",

    # Problèmes
    "mouchkil": "problème",
    "moshkil": "problème",
    "3dlina": "aide-moi",
    "mafhmt": "je n'ai pas compris",
    "fhamni": "explique-moi",
    "kayen mouchkil": "il y a un problème",

    # Quantité / intensité
    "bezzaf": "beaucoup",
    "bzaf": "beaucoup",
    "shwiya": "un peu",
    "ktar": "plus",
    "kther": "plus",
    "qalil": "peu",

    # Affirmations
    "sah": "c'est vrai",
    "mazel": "pas encore",
    "mashi": "ce n'est pas",
    "mahou": "ce n'est pas",

    # Directions
    "lmin": "à droite",
    "lysar": "à gauche",
    "lfo9": "en haut",
    "ltah": "en bas",
    "lota": "en bas",
    "l9oddam": "devant",
    "lwra": "derrière",

    # Activités
    "nkhdem": "je travaille",
    "ndrs": "j'étudie",
    "nktb": "j'écris",
    "nchof": "je vois",
    "nsawl": "je demande",
    "nabghi": "j'aime",

    # Pour ton défi Nuit de l'Info
    "waqtach nuit linfo": "quand est la nuit de l'info",
    "nuit linfo": "nuit de l'info",
    "defi": "défi",
    "challenge": "défi",
    "lo9t": "l'heure",
    "ssa3a": "l'heure",
    "makan": "lieu",
    "blasa": "lieu",
    "program": "programme",

    # Expressions courantes
    "ma3lich": "pas grave",
    "smahli": "excuse-moi",
    "allah y3awn": "que Dieu t'aide",
    "allah ykhalik": "s'il te plaît",
    "allah ybarek": "que Dieu te bénisse",
    "hadi": "cela",
    "hada": "ça",
    "houwa": "il",
    "hiya": "elle",

    # Fin conversation
    "bslama": "au revoir",
    "beslama": "au revoir",
    "yallah": "allons-y",
    "m3a salam": "au revoir",
}


# -------------------- UTILITAIRES TEXTES --------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\t\x0b\x0c]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_hassaniya(text: str) -> str:
    words = text.split()
    out = [HASSANIYA_DICTIONARY.get(w.lower(), w) for w in words]
    return " ".join(out)


def deduplicate_paragraphs(paragraphs: List[str], threshold: float = 0.9) -> List[str]:
    cleaned = [p for p in paragraphs if p and p.strip()]
    if not cleaned:
        return []
    vec = TfidfVectorizer().fit_transform(cleaned)
    sims = cosine_similarity(vec)
    keep = []
    used = set()
    for i, p in enumerate(cleaned):
        if i in used:
            continue
        keep.append(p)
        for j in range(i + 1, len(cleaned)):
            if sims[i, j] >= threshold:
                used.add(j)
    return keep

# -------------------- MONGODB HELPERS --------------------
def load_knowledge_from_db():
    try:
        doc = knowledge_collection.find_one({"_id": "current"})
        if doc and "paragraphs" in doc:
            paragraphs = doc["paragraphs"]
            if isinstance(paragraphs, list):
                st.session_state.knowledge_paragraphs = paragraphs
                rebuild_index()
    except Exception as e:
        st.sidebar.error(f"Erreur MongoDB au chargement : {e}")


def save_knowledge_to_db(paragraphs: List[str]):
    try:
        knowledge_collection.update_one(
            {"_id": "current"},
            {"$set": {"paragraphs": paragraphs, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
        st.sidebar.success("Base enregistrée dans MongoDB.")
    except Exception as e:
        st.sidebar.error(f"Erreur MongoDB à l'enregistrement : {e}")

# -------------------- INDEXATION --------------------
def rebuild_index():
    paragraphs = st.session_state.knowledge_paragraphs or []
    if not paragraphs:
        st.session_state.vectorizer = None
        st.session_state.tfidf_matrix = None
        st.session_state.bm25 = None
        return

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(paragraphs)

    st.session_state.vectorizer = vectorizer
    st.session_state.tfidf_matrix = tfidf_matrix

    if BM25_AVAILABLE:
        tokenized = [p.split() for p in paragraphs]
        st.session_state.bm25 = BM25Okapi(tokenized)
    else:
        st.session_state.bm25 = None

# -------------------- RECHERCHE PERTINENTE --------------------
def retrieve_relevant_paragraphs(question: str, top_k: int = 3, method: str = "bm25") -> List[Dict[str, Any]]:
    question = clean_text(question)
    question = normalize_hassaniya(question)

    paragraphs = st.session_state.knowledge_paragraphs or []
    if not paragraphs:
        return []

    results = []

    if method == "bm25" and st.session_state.bm25 is not None:
        tokens = question.split()
        scores = st.session_state.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        for idx in top_idx:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({"id": idx, "text": paragraphs[idx], "score": score})
        return results

    if st.session_state.vectorizer is None or st.session_state.tfidf_matrix is None:
        return []

    vec = st.session_state.vectorizer.transform([question])
    sims = cosine_similarity(vec, st.session_state.tfidf_matrix)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    for idx in top_indices:
        score = float(sims[idx])
        if score <= 0:
            continue
        results.append({"id": int(idx), "text": paragraphs[idx], "score": score})
    return results

# -------------------- CALL OLLAMA --------------------
def call_ollama(prompt: str, timeout: int = 40, model: str = OLLAMA_MODEL) -> str:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 256, "temperature": 0.2, "num_ctx": 2048},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = ""
        if isinstance(data, dict):
            text = data.get("response") or data.get("text") or data.get("output") or ""
            if not text:
                choices = data.get("choices") or data.get("result") or []
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        text = first.get("text") or first.get("content") or first.get("response") or ""
                    else:
                        text = str(first)
        if not text:
            text = str(data)
        return text.strip()
    except requests.Timeout:
        return "Erreur : le modèle local n'a pas répondu (timeout)."
    except Exception as e:
        return f"Erreur lors de l'appel au modèle local Ollama : {e}"

# -------------------- HISTORIQUE MULTI-TOUR --------------------
def build_conversation_history(max_turns: int = 6) -> str:
    if not st.session_state.messages:
        return "Aucun historique de conversation."
    recent = st.session_state.messages[-max_turns * 2 :]
    lines = []
    for msg in recent:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{role} : {content}")
    return "\n".join(lines)

# -------------------- PROMPT / GENERATION --------------------
def generate_answer(question: str, context_paragraphs: List[str], strict: bool = False) -> str:
    if not context_paragraphs:
        context_text = "Aucune information pertinente n'a été trouvée dans la base de connaissances."
    else:
        truncated = [p if len(p) <= 800 else p[:800] + "..." for p in context_paragraphs]
        context_text = "\n\n".join([f"[Paragraphe {i+1}]\n{p}" for i, p in enumerate(truncated)])

    history_text = build_conversation_history(max_turns=6)

    system_instructions = (
        "Tu es Rafiq-AI, un secrétaire virtuel pour le 'Défi national Nuit de l'Info 2025'.\n"
        "Tu réponds en français, de façon claire, concise et polie.\n"
        "Tu t'appuies uniquement sur les informations fournies dans le CONTEXTE (base de connaissances).\n"
        "Si le CONTEXTE ne contient pas la réponse à la question, tu dis explicitement que tu ne disposes pas de cette information dans la base actuelle.\n"
        "Tu peux comprendre quelques mots en arabe dialectal mauritanien (Hassaniya), mais tu réponds toujours en français.\n"
    )

    if strict:
        mode_text = (
            "Mode STRICT :\n"
            "- Ne réponds qu'avec des informations présentes dans le CONTEXTE.\n"
            "- Si une information n'est pas mentionnée dans le CONTEXTE, ne l'invente pas et dis que tu ne sais pas.\n"
        )
    else:
        mode_text = (
            "Mode INTELLIGENT :\n"
            "- Tu peux reformuler, structurer et clarifier les réponses.\n"
            "- MAIS tu ne dois pas inventer de nouveaux faits qui ne sont pas présents dans le CONTEXTE.\n"
        )

    prompt = f"""
{system_instructions}

HISTORIQUE DE LA CONVERSATION (multi-tour) :
{history_text}

CONTEXTE (base de connaissances) :
{context_text}

DERNIÈRE QUESTION DE L'UTILISATEUR :
{question}

Consignes générales :
1. Utilise l'historique pour comprendre les références comme "ça", "cette information", "comme je t'ai dit", etc.
2. Ne te contredis pas par rapport à tes réponses précédentes tant que le CONTEXTE ne change pas.
3. Répond uniquement à ce qui est lié au défi / à la base de connaissances.
4. Si tu ne trouves pas la réponse dans le CONTEXTE, dis-le clairement.

{mode_text}

Format de la réponse :
- Commence par une réponse courte et directe (1 à 2 phrases).
- Si c'est utile, ajoute ensuite des explications, exemples ou détails pratiques.
- Si tu ne sais pas, écris clairement : "Je ne dispose pas de cette information dans la base actuelle." et, si possible, propose à l'utilisateur d'ajouter le texte manquant dans la base de connaissances.

Réponds uniquement en français.
""".strip()

    return call_ollama(prompt)

# -------------------- PDF EXTRACTION --------------------
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        pages_text = []
        for _, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = clean_text(text)
            pages_text.append(text)
        full_text = "\n\n".join(pages_text)
        return full_text.strip()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF : {e}")
        return ""

# -------------------- WARMUP --------------------
def warmup_model():
    if st.session_state.model_warmed:
        return
    dummy_prompt = "Tu es un assistant. Dis simplement : OK."
    _ = call_ollama(dummy_prompt)
    st.session_state.model_warmed = True

# -------------------- UI STREAMLIT --------------------
st.set_page_config(page_title="Rafiq-AI (Amélioré)", page_icon="🤖", layout="wide")

# -------- SIDEBAR --------
with st.sidebar:
    st.title("Rafiq-AI — Admin")
    st.markdown("Assistant virtuel basé sur une base de connaissances (texte + PDF + JSON).")

    st.markdown("### État de la base")
    nb_paragraphs = len(st.session_state.knowledge_paragraphs)
    st.metric("Paragraphes en mémoire", nb_paragraphs)

    st.markdown("### Actions Base")
    if st.button("Recharger depuis MongoDB"):
        load_knowledge_from_db()
        st.rerun()

    if st.button("Vider la base (MongoDB)"):
        if st.session_state.knowledge_paragraphs:
            knowledge_collection.update_one(
                {"_id": "current"},
                {"$set": {"paragraphs": []}},
                upsert=True
            )
            st.session_state.knowledge_paragraphs = []
            rebuild_index()
            st.success("Base vidée et index mis à jour.")
            st.rerun()
        else:
            st.warning("La base est déjà vide.")

    st.markdown("---")

# -------- INIT --------
load_knowledge_from_db()
warmup_model()

# -------- LAYOUT PRINCIPAL --------
st.title("🤖 Rafiq-AI – Secrétaire virtuel du Défi national")
st.markdown("Colle du texte, importe un PDF ou un JSON, puis interagis avec le chatbot.")

col_knowledge, col_chat = st.columns([1.2, 2])

# -------- COLONNE GAUCHE : BASE --------
with col_knowledge:
    st.subheader(" Base de connaissances")

    # --- Ajout texte brut ---
    raw_text = st.text_area(
        "Texte à AJOUTER à la base (défi, services, FAQ…) :",
        height=200,
        placeholder="Colle ici un nouveau bloc de texte à ajouter à la base…",
    )

    # --- Ajout PDF ---
    uploaded_pdf = st.file_uploader(
        "Ou bien importe un PDF pour l'ajouter à la base :", type=["pdf"]
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Ajouter le texte"):
            if raw_text.strip():
                cleaned = clean_text(raw_text)
                cleaned = normalize_hassaniya(cleaned)
                parts = [p.strip() for p in re.split(r"\n{1,}|\r{1,}", cleaned) if p.strip()]
                if not parts:
                    parts = [cleaned]
                combined = st.session_state.knowledge_paragraphs + parts
                combined = deduplicate_paragraphs(combined, threshold=0.92)
                st.session_state.knowledge_paragraphs = combined
                save_knowledge_to_db(combined)
                rebuild_index()
                st.success(f"Texte ajouté. La base contient maintenant {len(combined)} paragraphe(s).")
            else:
                st.warning("Veuillez coller un texte avant d'ajouter.")

    with c2:
        if st.button("📎 Ajouter le PDF"):
            if uploaded_pdf is not None:
                pdf_text = extract_text_from_pdf(uploaded_pdf)
                if pdf_text:
                    pages = [p for p in pdf_text.split("\n\n") if p.strip()]
                    summarized = []
                    for i, p in enumerate(pages):
                        if not p.strip():
                            continue
                        short = p if len(p) < 2000 else p[:2000]
                        summary_prompt = (
                            "Tu es un assistant qui résume. Fais un résumé de la page en une phrase:\n\n" + short
                        )
                        summary = call_ollama(summary_prompt)
                        summarized.append(f"Page {i+1} summary: {summary}\n\n{p}")

                    combined = st.session_state.knowledge_paragraphs + summarized
                    combined = deduplicate_paragraphs(combined, threshold=0.92)
                    st.session_state.knowledge_paragraphs = combined
                    save_knowledge_to_db(combined)
                    rebuild_index()
                    st.success(f"PDF analysé et ajouté. La base contient maintenant {len(combined)} paragraphe(s).")
                else:
                    st.warning("Impossible d'extraire du texte depuis ce PDF.")
            else:
                st.warning("Veuillez sélectionner un fichier PDF avant de cliquer.")

    st.markdown("---")
    st.subheader(" Importer une base JSON")

    # --- Méthode 1 : fichier JSON ---
    uploaded_json = st.file_uploader("Importer un fichier JSON (liste de paragraphes)", type=["json"], key="json_file")
    if st.button("Charger le fichier JSON"):
        if uploaded_json is not None:
            try:
                data = json.load(uploaded_json)
                if isinstance(data, list):
                    # On nettoie un peu chaque paragraphe
                    cleaned_paras = [clean_text(str(p)) for p in data if str(p).strip()]
                    cleaned_paras = deduplicate_paragraphs(cleaned_paras, threshold=0.92)
                    st.session_state.knowledge_paragraphs = cleaned_paras
                    save_knowledge_to_db(cleaned_paras)
                    rebuild_index()
                    st.success(f"Base JSON importée. {len(cleaned_paras)} paragraphe(s) en mémoire.")
                    st.rerun()
                else:
                    st.error("Le fichier JSON doit contenir une LISTE de paragraphes (strings).")
            except Exception as e:
                st.error(f"JSON invalide : {e}")
        else:
            st.warning("Veuillez choisir un fichier JSON avant de cliquer.")

    # --- Méthode 2 : JSON collé ---
    json_text = st.text_area(
        "Ou collez ici une base JSON complète (liste de paragraphes) :",
        height=150,
        key="json_textarea",
        placeholder='Exemple : ["Paragraphe 1...", "Paragraphe 2...", "..."]'
    )

    if st.button("Importer la base JSON collée"):
        if json_text.strip():
            try:
                data = json.loads(json_text)
                if isinstance(data, list):
                    cleaned_paras = [clean_text(str(p)) for p in data if str(p).strip()]
                    cleaned_paras = deduplicate_paragraphs(cleaned_paras, threshold=0.92)
                    st.session_state.knowledge_paragraphs = cleaned_paras
                    save_knowledge_to_db(cleaned_paras)
                    rebuild_index()
                    st.success(f"Base JSON collée importée. {len(cleaned_paras)} paragraphe(s) en mémoire.")
                    st.rerun()
                else:
                    st.error("Le JSON doit être une LISTE de paragraphes (strings).")
            except Exception as e:
                st.error(f"JSON invalide : {e}")
        else:
            st.warning("Veuillez coller un contenu JSON avant d'importer.")

    # --- Méthode 3 : JSON via URL ---
    json_url = st.text_input("Ou bien URL d'une base JSON distante :", key="json_url")
    if st.button("Charger la base JSON depuis l'URL"):
        if json_url.strip():
            try:
                resp = requests.get(json_url.strip(), timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    cleaned_paras = [clean_text(str(p)) for p in data if str(p).strip()]
                    cleaned_paras = deduplicate_paragraphs(cleaned_paras, threshold=0.92)
                    st.session_state.knowledge_paragraphs = cleaned_paras
                    save_knowledge_to_db(cleaned_paras)
                    rebuild_index()
                    st.success(f"Base JSON chargée depuis l'URL. {len(cleaned_paras)} paragraphe(s) en mémoire.")
                    st.rerun()
                else:
                    st.error("Le JSON récupéré doit être une LISTE de paragraphes (strings).")
            except Exception as e:
                st.error(f"Erreur lors du chargement depuis l'URL : {e}")
        else:
            st.warning("Veuillez saisir une URL avant de charger.")

    st.markdown("---")
    if st.session_state.knowledge_paragraphs:
        with st.expander(" Voir & gérer les paragraphes enregistrés"):
            st.info("Tu peux supprimer, copier ou modifier chaque paragraphe.")
            for idx, p in enumerate(st.session_state.knowledge_paragraphs):
                st.markdown(f"**Paragraphe {idx}**")
                st.write(p)

                cols = st.columns([1, 1, 1])

                # Supprimer
                if cols[0].button(f"Supprimer {idx}", key=f"del_{idx}"):
                    paragraphs = st.session_state.knowledge_paragraphs.copy()
                    if 0 <= idx < len(paragraphs):
                        paragraphs.pop(idx)
                        st.session_state.knowledge_paragraphs = paragraphs
                        save_knowledge_to_db(paragraphs)
                        rebuild_index()
                        st.rerun()

                # Copier (manuel)
                if cols[1].button(f"Copier {idx}", key=f"copy_{idx}"):
                    st.info("Sélectionne manuellement le texte ci-dessus pour le copier.")

                # Modifier
                if cols[2].button(f"Modifier {idx}", key=f"edit_btn_{idx}"):
                    st.session_state[f"editing_{idx}"] = True

                if st.session_state.get(f"editing_{idx}", False):
                    st.markdown("_Édition de ce paragraphe :_")
                    edited_text = st.text_area(
                        f"Éditer le paragraphe {idx}",
                        value=p,
                        key=f"edit_text_{idx}",
                        height=120,
                    )
                    save_col, cancel_col = st.columns(2)
                    if save_col.button(f"Enregistrer {idx}", key=f"save_{idx}"):
                        paragraphs = st.session_state.knowledge_paragraphs.copy()
                        if 0 <= idx < len(paragraphs):
                            paragraphs[idx] = clean_text(edited_text)
                            st.session_state.knowledge_paragraphs = paragraphs
                            save_knowledge_to_db(paragraphs)
                            rebuild_index()
                            st.session_state[f"editing_{idx}"] = False
                            st.success(f"Paragraphe {idx} mis à jour.")
                            st.rerun()
                    if cancel_col.button(f"Annuler {idx}", key=f"cancel_{idx}"):
                        st.session_state[f"editing_{idx}"] = False
                        st.rerun()

    st.markdown("---")
    st.download_button(
        "Exporter la base (JSON)",
        data=json.dumps(st.session_state.knowledge_paragraphs, ensure_ascii=False, indent=2),
        file_name="knowledge_base.json",
    )

# -------- COLONNE DROITE : CHAT --------
with col_chat:
    st.subheader(" Chat avec Rafiq-AI")
    # historique
    for msg in st.session_state.messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        with st.chat_message(msg["role"]):
            st.markdown(f"**{role_label} :**")
            st.write(msg["content"])
        if msg["role"] == "assistant":
            st.markdown("---")

    strict_mode = st.checkbox("Mode STRICT (répondre uniquement selon la base)", value=True)

    user_prompt = st.chat_input("Écrivez votre question ici...")

    if user_prompt:
        question = user_prompt.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown("**User :**")
            st.write(question)

        qnorm = normalize_hassaniya(clean_text(question))

        retrieval_method = st.session_state.retrieval_method
        top_k = st.session_state.top_k

        method = "bm25" if ("bm25" in retrieval_method.lower() and st.session_state.bm25 is not None) else "tfidf"
        relevant = retrieve_relevant_paragraphs(qnorm, top_k=top_k, method=method)
        context_paragraphs = [r["text"] for r in relevant]

        # Paragraphes utilisés
        if relevant:
            with st.expander("Paragraphes utilisés pour répondre"):
                for r in relevant:
                    st.write(f"Paragraphe {r['id']} — score: {r['score']:.3f}")
                    st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
                    st.markdown("---")

        max_score = max([r["score"] for r in relevant], default=0.0)
        if max_score < st.session_state.similarity_threshold:
            assistant_text = "Je ne dispose pas de cette information dans la base actuelle."
        else:
            assistant_text = generate_answer(question, context_paragraphs, strict=strict_mode)

        assistant_text = assistant_text.strip()
        if len(assistant_text) > 2000:
            assistant_text = assistant_text[:2000] + "..."

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})

        with st.chat_message("assistant"):
            st.markdown("**Assistant :**")
            st.write(assistant_text)
            if relevant:
                st.markdown("**Sources :**")
                for r in relevant:
                    st.write(f"- Paragraphe {r['id']} (score {r['score']:.3f})")
            else:
                st.markdown("_Aucune source pertinente trouvée dans la base._")

        st.markdown("---")
