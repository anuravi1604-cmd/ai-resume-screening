import re
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

STOPWORDS = set(stopwords.words('english'))

COMMON_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "html", "css", "sql", "nosql", "react",
    "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot", "machine learning",
    "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "data analysis", "data vizualization", "aws",
    "azure", "gcp", "docker", "kubernetes", "git", "bash", "linux", "agile", "scrum", "jira",
    "communication", "teamwork", "leadership", "problem solving", "project management"
]

def extract_text_from_pdf(file):
    """
    Extract standard text from a PDF file object.
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()

def clean_text(text):
    """
    Clean resume text by removing HTML tags, urls, emails, special chars, and stop words.
    """
    # Convert to string just in case
    if not isinstance(text, str):
        text = str(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    clean_words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    
    return ' '.join(clean_words)

def extract_skills(text):
    """
    Extract predefined skills from the cleaned text.
    """
    extracted = []
    text_lower = text.lower()
    
    # Simple keyword matching
    for skill in COMMON_SKILLS:
        # We use negative lookarounds to avoid matching substrings within other words,
        # while correctly handling skills with non-alphanumeric chars like c++ or node.js
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text_lower):
            extracted.append(skill.title())
            
    return extracted
