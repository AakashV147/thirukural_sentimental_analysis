import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load dataset
df = pd.read_csv("/Users/aakashvenkatraman/Documents/thirukural_project/Thirukural With Explanation.csv")

print("Columns in dataset:", df.columns)

# Check if Translation column exists
if 'Translation' not in df.columns:
    raise KeyError("Column 'Translation' not found. Check CSV structure.")

# Drop missing values in Translation column
df = df.dropna(subset=['Translation'])

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Compute sentiment scores
df['sentiment_score'] = df['Translation'].apply(lambda text: sia.polarity_scores(str(text))['compound'])

# Categorize sentiment
def categorize_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['sentiment_score'].apply(categorize_sentiment)

print("Sentiment Distribution:\n", df['sentiment'].value_counts())

df.to_csv("thirukkural_with_sentiments.csv", index=False)

sns.countplot(data=df, x='sentiment', palette="coolwarm")
plt.title("Sentiment Analysis of Thirukkural")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

positive_text = " ".join(df[df['sentiment'] == 'Positive']['Translation'].dropna().astype(str))
negative_text = " ".join(df[df['sentiment'] == 'Negative']['Translation'].dropna().astype(str))

if positive_text:
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud_pos, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Positive Kurals")
    plt.show()

if negative_text:
    wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(negative_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud_neg, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Negative Kurals")
    plt.show()

sentiment_pipeline = pipeline("sentiment-analysis", framework="pt", device=0 if torch.cuda.is_available() else -1)

def get_transformer_sentiment(text):
    try:
        return sentiment_pipeline(str(text))[0]['label']
    except Exception as e:
        return "Error"

df['transformer_sentiment'] = df['Translation'].apply(get_transformer_sentiment)

df.to_csv("thirukkural_with_transformer_sentiments.csv", index=False)
print("Transformer-based sentiment analysis completed and saved.")
