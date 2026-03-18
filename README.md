```markdown
# Yatri Sathi Feedback Response System  

AI-powered automated feedback classification and response generation system built for  
**Webel (Department of IT & Electronics), Government of West Bengal**

---

## Overview  

The Yatri Sathi Feedback Response System is an end-to-end intelligent pipeline designed to process user feedback, classify it into predefined service categories, analyze sentiment, and generate context-aware automated responses.

The system combines traditional NLP techniques with large language models to improve customer support efficiency, ensure consistency, and enable scalable response generation.

---

## Key Features  

- Multilingual Support  
  - Language detection using AWS Comprehend  
  - Translation to English using AWS Translate (supports English, Hindi, Bengali)  

- Feedback Classification  
  - Uses TF-IDF and Cosine Similarity  
  - Categorizes feedback into service-specific buckets such as pricing, driver behavior, app issues, etc.  

- Sentiment Analysis  
  - Detects sentiment (Positive, Negative, Neutral) using AWS Comprehend  

- Automated Response Generation  
  - Uses Mixtral-8x7B via HuggingFace and LangChain  
  - Combines predefined templates with LLM-generated responses  

- Backend API  
  - Built using FastAPI for real-time processing  

- Frontend Interface  
  - Responsive UI built with HTML, CSS, and JavaScript  

---

## Architecture  

```

User Input (UI)
↓
FastAPI Backend
↓
Language Detection (AWS Comprehend)
↓
Translation (AWS Translate)
↓
TF-IDF + Cosine Similarity (Category Detection)
↓
Sentiment Analysis (AWS)
↓
LLM Response Generation (Mixtral via LangChain)
↓
Output Display (UI)

```

---

## Tech Stack  

### Backend  
- FastAPI  
- Python  

### AI / ML  
- AWS Comprehend  
- AWS Translate  
- TF-IDF (scikit-learn)  
- Cosine Similarity  
- LangChain  
- HuggingFace (Mixtral-8x7B)  

### Frontend  
- HTML  
- CSS  
- JavaScript  

### Libraries  
- boto3  
- pandas  
- scikit-learn  
- transformers  
- torch  
- faiss-cpu  

---

## Project Structure  

```

.
├── app.py                 # FastAPI backend
├── index.html            # Frontend UI
├── requirements.txt      # Dependencies
├── vector_database.json  # Generated keyword database
├── settings.py           # API keys (user-defined, not included)

````

---

## Installation and Setup  

### 1. Clone the Repository  

```bash
git clone https://github.com/your-username/yatri-sathi-feedback.git
cd yatri-sathi-feedback
````

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Configure Environment

Create a `settings.py` file in the root directory:

```python
HUGGINGFACEHUB_API_TOKEN = "your_token"
AWS_ACCESS_KEY_ID = "your_key"
AWS_SECRET_ACCESS_KEY = "your_secret"
```

---

### 4. Run the Application

```bash
python app.py
```

Open in browser:
[http://localhost:8000](http://localhost:8000)

---

## API Endpoint

### POST `/process_review/`

#### Request

```json
{
  "review_text": "Driver was rude and late"
}
```

#### Response

```json
{
  "response": {
    "user_response": "...",
    "feedback_category": "Driver Behaviour",
    "our_response_english": "..."
  }
}
```

---

## Example Workflow

1. User submits feedback:
   "Driver was rude and cab was late"

2. System performs:

   * Language detection and translation
   * Category classification
   * Sentiment analysis
   * Response generation

3. Output:

   * Structured feedback category
   * AI-generated response

---

## Highlights

* End-to-end automated feedback handling system
* Scalable and production-oriented design
* Achieved high classification accuracy with minimal misclassification
* Real-time processing using FastAPI
* Designed for a government-backed mobility platform

---

## Future Improvements

* Replace TF-IDF with embedding-based semantic search (FAISS + Sentence Transformers)
* Add analytics dashboard for feedback trends
* Extend multilingual support
* Improve response personalization
* Deploy as a cloud-based microservice

---

## Acknowledgements

* Webel (Department of IT & Electronics), Government of West Bengal
* AWS AI Services
* HuggingFace and LangChain ecosystem

---

## License  

This project was developed as part of an internship at Webel (Department of IT & Electronics), Government of West Bengal.  

The code is shared for educational and demonstration purposes only.  
Unauthorized commercial use or redistribution may be restricted.

```
```
