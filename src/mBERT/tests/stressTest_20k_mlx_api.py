import requests
import time
import random

def generate_random_text(base_text, index):
    return f"{base_text} - Message {index} - Random {random.randint(1, 10000)}"

def predict(text):
    url = 'http://127.0.0.1:8001/predict/'  # Update if your endpoint is different
    data = {"text": text, "model": "bert"}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Request failed"}

def main():
    # Generate unique sample texts
    base_text = "Sample SMS text"
    sample_texts = [generate_random_text(base_text, i) for i in range(20000)]

    # Stress test with progress logging
    start_time = time.time()

    for i, text in enumerate(sample_texts):
        prediction = predict(text)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} messages...")

    end_time = time.time()
    total_time = end_time - start_time
    total_messages = len(sample_texts)
    messages_per_second = total_messages / total_time

    print(f"Processed {total_messages} messages in {total_time:.2f} seconds")
    print(f"Throughput: {messages_per_second:.2f} messages per second")

if __name__ == '__main__':
    main()
