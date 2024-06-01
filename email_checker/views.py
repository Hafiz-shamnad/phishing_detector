import spacy
from django.conf import settings
from django.shortcuts import render
from .forms import EmailForm
from .predict import predict_email 

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in nlp.Defaults.stop_words]
    return ' '.join(tokens)


def home(request):
    form = EmailForm()
    prediction = None

    if request.method == 'POST':
        form = EmailForm(request.POST)
        if form.is_valid():
            email_text = form.cleaned_data['email_text']
            prediction = predict_email(email_text)
    
    return render(request, 'email_checker/home.html', {'form': form, 'prediction': prediction})
