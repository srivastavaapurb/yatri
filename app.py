import os
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import uvicorn
import nest_asyncio
import settings

# Apply nest_asyncio to allow running uvicorn in Colab
nest_asyncio.apply()

# Set environment variables
os.environ['HUGGINGFACEHUB_API_TOKEN'] = settings.HUGGINGFACEHUB_API_TOKEN

# Initialize AWS Comprehend client
comprehend_client = boto3.client('comprehend',
                                 region_name='us-west-2',
                                 aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)


# Initialize Langchain components
template = '''User response: {'review'} Our Response: Hi, we are from 'Yatri Sathi' (Official app for transport 
booking by Government of West Bengal). We are really sorry to hear your feedback. We are writing back to you in 
response to your feedback.'''

huggingface_hub = HuggingFaceHub(repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
                                 model_kwargs={'temperature': 0.235, 'max_length': 10})

prompt_template = PromptTemplate(template=template, input_variables={'review'},
                                 additional_instructions="**Company name is 'Yatri Sathi' not 'Ola' . So Answer "
                                                         "keeping this fact in mind.**** Also answer as per the user "
                                                         "feedback (i.e review_text)** Complete the sentence (don't "
                                                         "leave sentence abruptly) that addresses the user's review "
                                                         "in a way that every sentence ends in a full-stop. **Do not "
                                                         "mention any contact information such as email or website.**")

llm_chain = LLMChain(llm=huggingface_hub, prompt=prompt_template)

# Define FastAPI app
app = FastAPI()


class Review(BaseModel):
    review_text: str


def analyze_sentiment(review_text):
    response = comprehend_client.detect_sentiment(Text=review_text, LanguageCode='en')
    sentiment = response['Sentiment']
    return sentiment


def process_review(review_text):
    sentiment = analyze_sentiment(review_text)
    if sentiment == 'POSITIVE':
        return ("Our Response: Hi, we are from Yatri Sathi (official transport booking app by Govt. Of West Bengal). "
                "Thank you for your positive feedback. We are happy that you liked our service.")
    else:
        response = llm_chain.run(review_text)
        sentences = response.split('.')
        last_complete_sentence = '. '.join(sentences[:-1]) + '.'
        return last_complete_sentence


@app.post("/process_review/")
async def process_review_endpoint(review: Review):
    try:
        response = process_review(review.review_text)
        split_string = response.split("Our Response:")

        responses = {
            "user_response": split_string[0].replace("User response:", "").strip(),
            "our_response": split_string[1].strip()
        }

        return {"response": responses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run FastAPI app with Uvicorn in Colab
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
