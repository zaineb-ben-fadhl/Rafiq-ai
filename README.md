#  Rafiq-AI — Assistant Virtuel Intelligent
Rafiq-AI est un chatbot intelligent conçu pour le **Défi National Nuit de l'Info 2025**.  
Il s’adapte dynamiquement à une base de connaissances fournie par l’utilisateur (texte, PDF ou JSON) et répond en français aux questions liées au défi ou à toute autre information ajoutée.  

Ce projet utilise :
- **Ollama** (modèle local : Mistral)
- **Streamlit**
- **MongoDB**
- **TF-IDF / BM25** pour la recherche intelligente
- **Gestion du Hassaniya** (mini-traduction automatique)

---

##  Fonctionnalités principales

### 🔹 1. Base de connaissances dynamique
L'utilisateur peut :
- ajouter du texte,
- importer un PDF,
- importer un fichier JSON (option entreprise),
- modifier ou supprimer des paragraphes,
- exporter la base actuelle.

Rafiq-AI indexe automatiquement ces informations pour les utiliser dans ses réponses.

---

### 🔹 2. Chatbot IA avancé
- Comprend le **français** et les mots Hassaniya courants.
- Répond uniquement selon la base de connaissances en *Mode STRICT*.
- Peut reformuler en *Mode Intelligent*.
- Vision multi-tour : Rafiq-AI utilise l’historique pour comprendre le contexte.

---

### 🔹 3. Persistance via MongoDB
Toutes les données de la base de connaissances sont stockées dans MongoDB :

- conserve la base même après redémarrage
- utilisable sur plusieurs machines
- compatible avec MongoDB Atlas (cloud)

---

### 🔹 4. Explication des sources
Pour chaque réponse, Rafiq-AI affiche :
- les paragraphes utilisés,
- leurs scores de similarité.

Très utile pour :
- audit,
- transparence,
- présentations professionnelles.

---