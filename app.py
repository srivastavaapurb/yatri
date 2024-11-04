import os
import json
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from fastapi.responses import HTMLResponse
import uvicorn
import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Setting up environment
os.environ['HUGGINGFACEHUB_API_TOKEN'] = settings.HUGGINGFACEHUB_API_TOKEN

# Initialize AWS clients
comprehend_client = boto3.client('comprehend', region_name='us-west-2',
                                 aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)

translate_client = boto3.client('translate', region_name='us-west-2',
                                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)

# Predefined responses
predefined_responses = {
    'Price & Payment Related': 'We apologize for the inconvenience and are monitoring the situation. Please email details of the incident, including booking time and whether the issue is fare discrepancy or amount charged, to yatrisathi.support@wb.gov.in. Note that charges vary with demand at that time. For fare discrepancies, we will review the route with the driver.',
    'Cab Availability & Wait time': 'We regret the incident. We continuously monitor cab availability in our service areas and communicate with drivers regarding delays. Please note that cab availability is lower during peak hours, so we keep a close watch on this. For sharing details of the incident, please email us at yatrisathi.support@wb.gov.in or call us at 03344400033.',
    'Driver Behaviour': 'We apologize for the incident and are speaking with the driver to understand their perspective. We are also reviewing their history for any past issues; if necessary, appropriate actions, including suspension, will be taken. For first-time incidents, drivers receive a warning to uphold a positive customer experience. Feel free to email us at yatrisathi.support@wb.gov.in or call 03344400033 to share further details',
    'App related (Technical)': 'We apologize for this app related issue. We would be grateful if you can share the details of the issue with us at yatrisathi.support@wb.gov.in, so that we can share this with our team working in the backend. We would like to thank you for your patience and we request you to wait for one or two hours more so that we can fix the issue. We would like to ensure you that such things won\'t happen in the future again',
    'Special Zone Issues': 'We apologize for the incident. Please email us the details at yatrisathi.support@wb.gov.in, including whether you took a cab from a queue despite booking through the app. We want to inform you that certain locations are often overcrowded, causing cab availability issues. Rest assured, we are working on resolving this to prevent queue-related problems in the future',
    'Lost & Found': 'We would request you not to worry if you have accidently left your belongings in the car. In such cases the drivers report us immediately. You will surely get it back as we have best drivers who will definitely return your belongings with care and you can even communicate with the driver regarding this matter. For any further issues feel free to contact us at yatrisathi.support@wb.gov.in or 03344400033',
    'Cab Cleanliness': 'We apologize for the inconvenience caused due to the cleanliness of the cab. We strive to maintain the highest standards of hygiene and will address this issue with the driver concerned. Kindly share more details with us at yatrisathi.support@wb.gov.in so we can ensure such incidents do not occur in the future',
    'Others': ''
}

# Extract keywords and important lines categorized by feedback category
vector_db = {category: {'keywords': [], 'important_lines': []} for category in predefined_responses}

for category, response in predefined_responses.items():
    response_lines = response.split('. ')
    for line in response_lines:
        vector_db[category]['keywords'].extend(line.split())
        vector_db[category]['important_lines'].append(line)

# Save categorized keywords and important lines to a JSON file
with open('vector_database.json', 'w') as f:
    json.dump(vector_db, f)

# Expanded keyword dictionary
categories = {
    'Price & Payment Related': ['rate', 'price', 'prices', 'money', 'payment', 'charge', 'cost', 'fare', 'bill', 'refund', 'wallet',
                                'cheaper', 'cheap'],
    'Cab Availability & Wait time': ['wait time', 'availability', 'booking', 'available', 'delay', 'late', 'timing', 'schedule'],
    'Driver Behaviour': ['driver', 'behaviour', 'attitude', 'rude', 'polite', 'professional', 'behavior', 'behaved',
                         'manners', 'drivers', 'guy','driver.','driver\'s'],
    'Cab Cleanliness': ['clean', 'dirty', 'hygiene', 'smell', 'tidy', 'neat', 'sanitary', 'cleanliness','untidy','dirt'],
    'App related (Technical)': ['app', 'login', 'technical', 'Spam','spam','issue', 'bug', 'crash', 'error', 'problem', 'glitch'],
    'Special Zone Issues': ['zone', 'area', 'region', 'place', 'spot', 'site','queue'],
    'Lost & Found': ['lost', 'found', 'left', 'item', 'belonging', 'possession', 'article'],
    'Others': []
}

# Update the template to use important lines from the vector database as part of the prompt
template_general = '''YOUR FEEDBACK: {review}; OUR RESPONSE: We are really sorry to hear your feedback. {important_lines}'''
template_app = '''YOUR FEEDBACK: {review}; OUR RESPONSE: We are sorry to hear your feedback. {important_lines}'''

# Initialize language model and template
huggingface_hub = HuggingFaceHub(repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
                                 model_kwargs={'temperature': 0.407, 'max_length': 500})

prompt_template_general = PromptTemplate(template=template_general, input_variables=['review', 'important_lines'])
prompt_template_app = PromptTemplate(template=template_app, input_variables=['review', 'important_lines'])

llm_chain_general = LLMChain(llm=huggingface_hub, prompt=prompt_template_general)
llm_chain_app = LLMChain(llm=huggingface_hub, prompt=prompt_template_app)

app = FastAPI()


class Review(BaseModel):
    review_text: str


def detect_language(text):
    try:
        response = comprehend_client.detect_dominant_language(Text=text)
        languages = response['Languages']
        if not languages:
            return 'UNKNOWN'
        return languages[0]['LanguageCode']
    except Exception as e:
        print(f"Error detecting language: {e}")
        return 'UNKNOWN'


def translate_text(text, source_lang='auto', target_lang='en'):
    try:
        response = translate_client.translate_text(Text=text, SourceLanguageCode=source_lang,
                                                   TargetLanguageCode=target_lang)
        return response['TranslatedText']
    except Exception as e:
        print(f"Error translating text: {e}")
        return text


def analyze_sentiment(review_text):
    response = comprehend_client.detect_sentiment(Text=review_text, LanguageCode='en')
    sentiment = response['Sentiment']
    return sentiment


def categorize_feedback(review_text):
    # Tokenize review text
    review_tokens = review_text.lower().split()

    # Create a DataFrame with categories and their associated keywords
    category_keywords = []
    for category, keywords in categories.items():
        for keyword in keywords:
            category_keywords.append((category, keyword))

    category_keywords_df = pd.DataFrame(category_keywords, columns=['category', 'keyword'])

    # Calculate TF-IDF for the keywords
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(category_keywords_df['keyword'])

    # Calculate TF-IDF for the review text
    review_tfidf = tfidf_vectorizer.transform([" ".join(review_tokens)])

    # Calculate cosine similarity between review text and keywords
    cosine_similarities = cosine_similarity(review_tfidf, tfidf_matrix)

    # Aggregate similarity scores by category
    category_scores = {}
    for idx, (category, keyword) in enumerate(category_keywords):
        if category not in category_scores:
            category_scores[category] = 0
        category_scores[category] += cosine_similarities[0, idx]

    # Find the best matching category
    if max(category_scores.values()) < 0.1:
        return 'Others'

    best_category = max(category_scores, key=category_scores.get)

    return best_category


def is_valid_feedback(detected_language):
    # Check if the detected language is one of the allowed languages
    allowed_languages = ['en', 'hi', 'bn']  # English, Hindi, Bengali
    return detected_language in allowed_languages


def process_review(review_text):
    # Check for specific inputs 'Aa' and 'ok'
    if review_text.lower() == 'aa':
        return "undefined response"

    if review_text.lower() == 'ok':
        category = 'Others'
        sentiment = 'POSITIVE'
        review_text = 'ok'
    else:
        detected_language = detect_language(review_text)

        if not is_valid_feedback(detected_language):
            return "Please enter valid feedback"

        if detected_language != 'en':
            review_text = translate_text(review_text, source_lang=detected_language, target_lang='en')

        category = categorize_feedback(review_text)
        sentiment = analyze_sentiment(review_text)

    with open('vector_database.json', 'r') as f:
        vector_db = json.load(f)

    # Select important lines related to the category
    if category == 'Others':
        important_lines = "We want to address the issues faced by you. Our contact email is yatrisathi.support@wb.gov.in"
    else:
        important_lines = ". ".join(line for line in vector_db[category]['important_lines'])

    if sentiment == 'POSITIVE' and category == 'App related (Technical)':
        response_english = (
            f"Hi, we are from Yatri Sathi (official e-mobility booking app by"
            f" Govt. Of West Bengal). Thank you for your positive app related feedback. We are glad that you liked Yatri Sathi."
        )
    elif sentiment == 'POSITIVE' and category == 'Others':
        response_english = (
            f"Hi, we are from Yatri Sathi (official e-mobility booking app by"
            f" Govt. Of West Bengal). Thank you for your positive feedback. We are glad that you liked Yatri Sathi."
        )
    elif sentiment == 'POSITIVE':
        response_english = (
            f"Hi, we are from Yatri Sathi (official e-mobility booking app by"
            f" Govt. Of West Bengal). Thank you for your positive {category.lower()} feedback. We are glad that you liked Yatri Sathi."
        )
    else:
        if category == 'Price & Payment Related':
            response_english = llm_chain_general.run(review=review_text, important_lines=important_lines)
        elif category == 'App related (Technical)':
            response_english = llm_chain_app.run(review=review_text, important_lines=important_lines)
        else:
            response_english = llm_chain_general.run(review=review_text, important_lines=important_lines)

    return (
        f"YOUR FEEDBACK: {review_text}\n"
        f"FEEDBACK CATEGORY: {category}\n"
        f"OUR RESPONSE in English: {response_english}\n"
    )


@app.post("/process_review/")
async def process_review_endpoint(review: Review):
    try:
        response = process_review(review.review_text)
        if response == "Please enter valid feedback" or response == "undefined response":
            return {"response": response}
        split_string = response.split("\n")
        responses = {
            "user_response": split_string[0].replace("YOUR FEEDBACK:", "").strip(),
            "feedback_category": split_string[1].replace("FEEDBACK CATEGORY:", "").strip(),
            "our_response_english": split_string[2].replace("OUR RESPONSE in English:", "").strip()
        }
        return {"response": responses}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
