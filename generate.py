import csv
import random
from faker import Faker

fake = Faker()

# Sentiment distribution (33% Positive, 33% Neutral, 34% Negative)
sentiments = [1] * 3300 + [0] * 3300 + [-1] * 3400
random.shuffle(sentiments)

# Emotion pools
emotions = {
    1: ["happy", "excited", "grateful", "joyful", "peaceful"],
    0: ["indifferent", "bored", "calm", "relaxed"],
    -1: ["angry", "sad", "frustrated", "annoyed"]
}

# Generate posts
posts = []
for _ in range(10000):
    sentiment = random.choice(sentiments)
    platform = random.choice(["Twitter", "Instagram"])
    emotion = random.choice(emotions[sentiment])
    
    # Generate text based on sentiment
    if sentiment == 1:
        text = f"{fake.sentence()} {random.choice(['😊', '🌟', '🎉'])} #{fake.word().capitalize()}{fake.word().capitalize()}"
    elif sentiment == 0:
        text = f"{fake.sentence()} {random.choice(['😐', '🛋️', '🌧️'])} #{fake.word().capitalize()}"
    else:
        text = f"{fake.sentence()} {random.choice(['😤', '👎', '😩'])} #{fake.word().capitalize()}Fail"
    
    posts.append([platform, text, sentiment, emotion])

# Save to CSV
with open("social_media_posts_10k.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["platform", "text", "sentiment", "emotion"])
    writer.writerows(posts)