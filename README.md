# Rafiq-AI — Assistant Virtuel Intelligent pour la Nuit de l’Info 2025

Rafiq-AI est un **chatbot intelligent** conçu pour le **Défi National – Nuit de l'Info 2025**.  
Il s’adapte dynamiquement à une base de connaissances fournie par l’utilisateur (texte, PDF ou JSON) et répond **en français**, avec une gestion basique du **Hassaniya**, aux questions liées au défi ou à toute autre information ajoutée.

🎥 **Démo vidéo** :  
https://drive.google.com/file/d/1-brKbK3roqyz_4UOcnn_DceB99lJKndb/view?usp=sharing

---

## 🚀 Stack technique

Le projet utilise :

- **Python 3.10+**
- **Streamlit** – Interface web simple et rapide
- **Ollama** – Modèle de langage local (`mistral`)
- **MongoDB** – Stockage persistant de la base de connaissances
- **Scikit-learn (TF-IDF)** et **BM25** – Recherche sémantique des meilleurs passages
- Gestion basique de **Hassaniya** (mots clés / mini-traduction)

---

## ✨ Fonctionnalités principales

### 🔹 1. Base de connaissances dynamique

L’utilisateur peut :

- ajouter du texte libre,
- importer un **fichier PDF**,
-  importer un **URL MONGO**,
- importer un **fichier JSON** (cas entreprise : FAQ, documentation interne…),
- modifier ou supprimer des paragraphes,
- exporter la base actuelle.

Rafiq-AI indexe automatiquement ces informations (TF-IDF / BM25) pour les réutiliser dans ses réponses.

---

### 🔹 2. Chatbot IA avancé

- Comprend le **français** et quelques mots clés en **Hassaniya**.
- **Mode STRICT** : répond uniquement à partir de la base de connaissances (pas d’invention).
- **Mode INTELLIGENT** : reformule, résume et enrichit la réponse tout en restant aligné sur les sources.
- Gère le **contexte multi-tour** : l’historique de la conversation est pris en compte.

---

### 🔹 3. Persistance des données (MongoDB)

Tous les contenus de la base de connaissances sont stockés dans **MongoDB** :

- la base reste disponible après redémarrage,
- plusieurs utilisateurs/machines peuvent se connecter à la même base,
- compatible **MongoDB local** ou **MongoDB Atlas (cloud)**.

---

### 🔹 4. Traçabilité et transparence

Pour chaque réponse, Rafiq-AI peut afficher :

- les paragraphes utilisés,
- leurs scores de similarité (TF-IDF / BM25),
- la source (texte manuel / PDF / JSON).

Utile pour :

- l’audit des réponses,
- la transparence des décisions,
- les présentations professionnelles et démonstrations.

