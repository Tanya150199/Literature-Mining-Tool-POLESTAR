# Polestar Literature Mining Tool

This README provides comprehensive instructions on setting up and using the Polestar Literature Mining Tool, a web application designed to help researchers efficiently extract and analyze scientific papers from databases like PubMed and DOAJ. This tool leverages natural language processing and machine learning techniques to enhance the accuracy and interpretability of research article searches and analysis.

## Prerequisites

Before beginning, ensure your system has Python installed. If Python is not installed, follow these steps to install it.

### Installing Python

1. **Download Python:**
   Visit the official Python website at [python.org](https://python.org) and download the latest version of Python for your operating system.

2. **Install Python:**
   Open the downloaded file and follow the installation instructions.

3. **Verify Installation:**
   Open a terminal and type `python --version` or `python3 --version` to check the installed version.

## Setting Up the Literature Mining Tool Application 

1. **Clone the repository:**

   ```bash
   git clone https://your-repository-url.git
   cd your-repository-directory

2. **Create virtual environment (optional but recommended):**

This step is optional but recommended to keep your project's dependencies separate from the global Python environment.

    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Required Libraries**

The application requires several libraries. Install them using pip:
pip install Flask requests pandas nltk scikit-learn transformers

After installing the libraries, you need to download additional NLTK data:

    python -m nltk.downloader stopwords wordnet punkt

4. **Clone the Repository**
If you have your code in a Git repository:

    git clone https://your-repository-url.git
    cd your-repository-directory

If not, ensure your files are structured as follows in your project directory:

/Polestar_Literature_Tool
    /static
        - polestar_logo.png  # Place your logo image here
    /templates
        - index.html  # HTML template for the web interface
    app.py
    literature_mining.py

5. Running the Application

    1. Start the server:

    Ensure you are in the project root directory (where app.py is located) and run:

        python app.py

    This command starts the Flask development server on http://127.0.0.1:5000/.

    2. Access the application:

    Open a web browser and navigate to http://127.0.0.1:5000/ to use the application.

## Usage

1. Search for Articles:

Use the web form to enter a search query, select the number of results, and choose the database source (PubMed or DOAJ). Click 'Search' to retrieve and display the results.

2. Answer Questions:

Provide a question and context in the respective fields of the web form. Click 'Get Answer' to have the system process the question and return an answer based on the provided context.

## Troubleshooting

1. Ensure all commands are run in the virtual environment if you've set one up.
2. Verify that Flask is running without errors by checking the terminal output.