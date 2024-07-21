import os
import nltk
from flask import Flask, render_template, request, jsonify
from literature_mining import literature_mining_tool, answer_question

application = Flask(__name__)  # Rename app to application

def download_nltk_data():
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    nltk.data.path.append(nltk_data_path)
    
    nltk_packages = ['stopwords', 'wordnet', 'punkt']
    for package in nltk_packages:
        try:
            nltk.data.find(f'corpora/{package}')
        except LookupError:
            nltk.download(package, download_dir=nltk_data_path)

@application.before_first_request
def setup():
    download_nltk_data()

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    max_results = request.form.get('max_results', 10, type=int)
    source = request.form.get('source', 'pubmed')  # Default to 'pubmed' if not provided
    if query:
        results_df = literature_mining_tool(query, max_results, source)
        if 'Keywords' in results_df.columns:
            results_df = results_df.drop(columns=['Keywords'])
        results_html = results_df.to_html(classes='table table-striped', index=False, escape=False)
        return jsonify({'results': results_html})
    return jsonify({'error': 'No query provided'})

@application.route('/answer', methods=['POST'])
def answer():
    question = request.form['question']
    context = request.form['context']
    if question and context:
        answer_data = answer_question(question, context)
        if 'answer' not in answer_data or 'score' not in answer_data:
            answer_data = {'answer': 'Could not find an answer.', 'score': 0}
        return jsonify(answer_data)
    return jsonify({'error': 'Question or context not provided'})

if __name__ == '__main__':
    download_nltk_data()
    application.run(debug=True)
