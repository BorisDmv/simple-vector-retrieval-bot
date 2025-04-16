# Simple Vector Retrieval Chatbot

This project is a simple semantic retrieval-based chatbot that uses word embeddings to find the most relevant answers based on user input. It loads a pre-trained **Word2Vec** model, processes the user's question, and compares it with pre-defined questions to return the best match. If no good match is found, it provides a default fallback response.

## Features

- **Word2Vec** embeddings: Uses the pre-trained **Word2Vec Google News 300** model for semantic similarity.
- **CSV-based training data**: Questions and corresponding answers are stored in a CSV file.
- **Simple API**: Exposes a REST API (`/chat`) to interact with the bot.
- **Cosine similarity**: Measures the similarity between user input and pre-defined questions using cosine similarity.
- **Fallback responses**: Provides a default response when the input doesn't match any trained question closely enough.

## Requirements

- Python 3.6+
- `gensim` library (for Word2Vec model)
- `sklearn` (for cosine similarity)
- `flask` (for the API server)
- `pandas` (for CSV file handling)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/BorisDmv/simple-vector-retrieval-bot.git
   cd simple-vector-retrieval-bot
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This is if you installed on homebrew on macos:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. Download the pre-trained **Word2Vec** model. The code uses the **Google News 300** model, but you can choose any other pre-trained model if needed:
   ```bash
   import gensim.downloader as api
   word_model = api.load("word2vec-google-news-300")
   ```

## How It Works

The chatbot works by embedding both the user input and pre-defined questions into vector space using **Word2Vec** embeddings. Then it calculates the cosine similarity between the user's input vector and each question vector. The question with the highest similarity score is selected, and its corresponding answer is returned.

### Example Workflow:

1. **User Input**: "What time is it?"
2. **Chatbot Processing**: The chatbot tokenizes and embeds the input, then compares it with all pre-defined questions.
3. **Cosine Similarity Calculation**: The input's embedding is compared with the embeddings of the pre-defined questions.
4. **Answer**: The question with the highest cosine similarity is selected, and its corresponding answer is returned.

If the similarity score is below a defined threshold, the bot will reply with:

## API

The chatbot exposes a simple API that can be interacted with through POST requests.

### POST `/chat`

#### Request

```json
{
  "user_input": "user's question"
}

Response

{
  "response": "The bot's answer"
}

curl -X POST -H "Content-Type: application/json" -d '{"user_input": "Hello"}' http://localhost:5000/chat

```
