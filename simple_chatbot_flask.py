from flask import Flask, request, jsonify
import csv
import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import TreebankWordTokenizer

# Initialize the Flask app
app = Flask(__name__)

# Load Word2Vec model
word_model = api.load("word2vec-google-news-300")
tokenizer = TreebankWordTokenizer()

# Preprocess function
def preprocess(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    return [token for token in tokens if token.isalnum() and token in word_model.key_to_index]

# Get text embedding function
def get_text_embedding(text_tokens, model):
    embeddings = [model[token] for token in text_tokens if token in model.key_to_index]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

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

csv_file_path = "scanlab_training_data.csv"  # Adjust this if your CSV is elsewhere
training_questions, training_answers = load_qa_pairs(csv_file_path)

# Preprocess and embed training questions
processed_questions = [preprocess(q) for q in training_questions]
question_embeddings = np.array([get_text_embedding(tokens, word_model) for tokens in processed_questions])

# Chatbot response function
def respond(user_input, questions, answers, question_embeddings, word_model, similarity_threshold=0.8):
    processed_input = preprocess(user_input)

    if not processed_input:
        return "Please say something meaningful."

    input_embedding = get_text_embedding(processed_input, word_model)

    if np.all(input_embedding == 0):
        return "Sorry, I don't understand the words you used."

    similarities = cosine_similarity(input_embedding.reshape(1, -1), question_embeddings)[0]
    most_similar_index = np.argmax(similarities)
    highest_similarity = similarities[most_similar_index]

    if highest_similarity >= similarity_threshold:
        return answers[most_similar_index]
    else:
        return "Sorry, I'm not sure how to respond to that."

# API route to handle chat
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"response": "No input provided"}), 400
    
    response = respond(user_input, training_questions, training_answers, question_embeddings, word_model)
    return jsonify({"response": response})

# Start the Flask server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
