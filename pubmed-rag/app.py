from flask import Flask, request, jsonify, render_template
from services.pubmed import fetch_pubmed_articles
from services.vectorstore import create_or_load_db
from services.rag import ask_medical_bot

app = Flask(__name__)

vectorstore_cache = {}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    try:
        if topic not in vectorstore_cache:
            print(f"🚀 New topic: {topic}")

            articles = fetch_pubmed_articles(topic)

            if not articles:
                return jsonify({"error": "No articles found on PubMed."}), 404

            vector_db = create_or_load_db(topic, articles)
            if not vector_db:
                return jsonify({"error": "Failed to create vector DB."}), 500

            vectorstore_cache[topic] = vector_db
        else:
            print("Using cached vectorstore")

        db = vectorstore_cache[topic]
        answer = ask_medical_bot(topic, db)

        return jsonify({
            "topic": topic,
            "response": answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    topic = data.get("topic", "").strip()
    question = data.get("question", "").strip()

    if not topic or not question:
        return jsonify({"error": "Both topic and question are required."}), 400

    if topic not in vectorstore_cache:
        return jsonify({"error": "Search topic first."}), 400

    try:
        db = vectorstore_cache[topic]
        answer = ask_medical_bot(question, db)

        return jsonify({
            "topic": topic,
            "question": question,
            "response": answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    vectorstore_cache.clear()
    return jsonify({"message": "Cache cleared successfully."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)