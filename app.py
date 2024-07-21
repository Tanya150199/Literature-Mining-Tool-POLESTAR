import nltk
from flask import Flask, render_template, request, jsonify
from literature_mining import literature_mining_tool, answer_question

app = Flask(__name__)

@app.before_first_request
def setup():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
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

@app.route('/answer', methods=['POST'])
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
    app.run(debug=True)
