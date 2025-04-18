from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "https://localhost:8080"}})

# Load Word2Vec model
print("Loading Word2Vec model (this may take some time)...")
word_model = api.load("word2vec-google-news-300")
print("Word2Vec model loaded!")

tokenizer = TreebankWordTokenizer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    return [token for token in tokens if token.isalnum() and token in word_model.key_to_index]

# Load Q&A pairs from CSV
def load_qa_pairs(csv_path):
    questions = []
    answers = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            questions.append(row['question'])
            answers.append(row['answer'])
    return questions, answers

csv_file_path = "scanlab_training_data.csv"
training_questions, training_answers = load_qa_pairs(csv_file_path)

# Fit TF-IDF on training questions
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(training_questions)

# Get TF-IDF weighted text embedding
def get_weighted_embedding(text, model, tfidf_vectorizer):
    tokens = preprocess(text)
    if not tokens:
        return np.zeros(model.vector_size)

    tfidf_weights = tfidf_vectorizer.transform([text]).toarray()[0]
    vocab = tfidf_vectorizer.get_feature_names_out()
    word_weights = {vocab[i]: tfidf_weights[i] for i in range(len(vocab))}

    vectors = []
    for word in tokens:
        if word in model.key_to_index:
            weight = word_weights.get(word, 0.0)
            vectors.append(model[word] * weight)

    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Precompute embeddings for training questions
question_embeddings = np.array([
    get_weighted_embedding(q, word_model, tfidf_vectorizer)
    for q in training_questions
])

# Chatbot response function
def respond(user_input, questions, answers, question_embeddings, word_model, tfidf_vectorizer, similarity_threshold=0.6):
    input_embedding = get_weighted_embedding(user_input, word_model, tfidf_vectorizer)

    if np.all(input_embedding == 0):
        return "Sorry, I don't understand the words you used."

    similarities = cosine_similarity(input_embedding.reshape(1, -1), question_embeddings)[0]
    most_similar_index = np.argmax(similarities)
    highest_similarity = similarities[most_similar_index]

    if highest_similarity >= similarity_threshold:
        return answers[most_similar_index]
    else:
        return "Sorry, I'm not sure how to respond to that."

# Flask route
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"response": "No input provided"}), 400

    response = respond(user_input, training_questions, training_answers, question_embeddings, word_model, tfidf_vectorizer)
    return jsonify({"response": response})

# Run server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)
