import os
import json
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.secret_key = 'supersecretkey'
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Function to check for duplicate or similar questions
def check_duplicate_questions(questions):
    vectorizer = TfidfVectorizer().fit_transform(questions)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    duplicates = []
    similar_questions = []
    duplicate_threshold = 1.0  # Cosine similarity threshold to mark as duplicate
    similar_threshold = 0.8 # Cosine similarity threshold to mark as similar
    for i in range(len(cosine_matrix)):
        for j in range(i+1, len(cosine_matrix)):
            if cosine_matrix[i][j] > similar_threshold and cosine_matrix[i][j] < duplicate_threshold:
                similar_questions.append({
                    'question1': questions[i],
                    'question2': questions[j],
                    'similarity': cosine_matrix[i][j]
                })
            elif cosine_matrix[i][j] >= duplicate_threshold:
                duplicates.append({
                    'question1': questions[i],
                    'question2': questions[j],
                    'similarity': cosine_matrix[i][j],
                })

    return duplicates, similar_questions

# def paraphrase_question(question_text):
#     # Call GPT-4 API
#     response = openai.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a question paraphraser."},
#         {
#             "role": "user",
#             "content": question_text
#         }
#     ],
#     temperature=0.7,
#     max_tokens=100
# )
    
    # Extract the paraphrased text
    # paraphrased_question = response['choices'][0]['text'].strip()
    # return paraphrased_question

# Function to extract questions from HTML
def extract_questions(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    questions = []
    for p in soup.find_all('p'):
        text = p.get_text().strip()
        if text.endswith('?'):
            questions.append(text)
    return questions

# Function to classify questions using Bloom's Taxonomy
def classify_bloom_taxonomy(question):
    bloom_keywords = {
        'Remember': ['define', 'list', 'recall', 'name', 'identify'],
        'Understand': ['describe', 'explain', 'paraphrase', 'summarize'],
        'Apply': ['apply', 'use', 'solve', 'demonstrate', 'illustrate'],
        'Analyze': ['analyze', 'compare', 'contrast', 'distinguish', 'examine'],
        'Evaluate': ['evaluate', 'judge', 'critique', 'justify', 'defend'],
        'Create': ['create', 'design', 'develop', 'formulate', 'construct']
    }
    
    question_lower = question.lower()
    for level, keywords in bloom_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            return level
    return 'Unclassified'

# Function to process text and extract nouns
def process_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    pos_tags = pos_tag(cleaned_tokens)
    nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
    return nouns

# New function to handle JSON-based question banks
def process_json(question_bank):
    questions = []
    for question in question_bank.get('questions', []):
        questions.append(question.get('question', ''))
    return questions

# Function to generate wordcloud
def generate_wordcloud(questions):
    # Combine all questions into a single string
    all_words = ' '.join(questions)

    # Tokenize the combined string into words
    tokens = word_tokenize(all_words)

    # Filter out stopwords and keep only nouns (NN*)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word, pos in pos_tag(tokens) 
                      if pos.startswith('NN') and word.lower() not in stop_words]

    # Ensure there are words to create a word cloud
    if len(filtered_words) == 0:
        raise ValueError("No nouns found in the text to generate a word cloud.")

    # Generate the word cloud from filtered words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))

    # Save the word cloud image
    wordcloud.to_file("static/wordcloud.png")

    return wordcloud

# Function to handle file uploads
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    # Check if the file is JSON or HTML
    content_type = file.content_type
    if content_type == 'application/json':
        # Handle JSON file upload
        content = file.read().decode('utf-8')
        question_bank = json.loads(content)
        questions = process_json(question_bank)
        print(questions)
    elif content_type == 'text/html':
        # Handle HTML file upload
        content = file.read().decode('ISO-8859-1')
        questions = extract_questions(content)
    else:
        return "Unsupported file format", 400

    # Proceed with noun extraction and word cloud creation
    word_freq = defaultdict(lambda: defaultdict(int))
    for i, question in enumerate(questions, 1):
        nouns = process_text(question)
        for noun in set(nouns):
            word_freq[noun][f'Q{i}'] += 1
    
    total_freq = {word: sum(freq.values()) for word, freq in word_freq.items()}

    wordcloud = generate_wordcloud(questions)
    wordcloud.to_file("static/wordcloud.png")
    
    sorted_words = sorted(total_freq.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_words[:20])  # Top 20 words
    
    # Tagging questions and Bloom's taxonomy analysis
    tagged_questions = []
    for i, question in enumerate(questions, 1):
        tags = [word for word in top_words if word in question.lower()]
        bloom_category = classify_bloom_taxonomy(question)
        tagged_questions.append({
            'id': f'Q{i}',
            'question': question,
            'tags': tags,
            'bloom': bloom_category
        })

    duplicates, similar_questions = check_duplicate_questions(questions)

    # Render the results
    return render_template("analysis.html", 
                           wordcloud_image="/static/wordcloud.png", 
                           word_freq=word_freq,
                           top_words=top_words,
                           tagged_questions=tagged_questions,
                           duplicates=duplicates,
                           similar_questions=similar_questions)

if __name__ == '__main__':  
    # questions = [
    #      "What is the capital of France?",
    #      "Explain the concept of gravity.",
    #      "Describe the process of photosynthesis."
    # ]

    # try:
    #      generate_wordcloud(questions)
    #      print("Word cloud generated successfully!")
    # except ValueError as e:
    #      print(e)

    app.run(debug=True)
