import json
import requests

# Load test samples
with open('test_samples.json', 'r') as f:
    samples = json.load(f)

local_url = "http://localhost:8002/predict/"
external_url = "https://europe.ots-api.telecomsxchange.com/predict/"

results = []

for sample in samples:
    text = sample['text']
    expected = sample['expected']
    category = sample['category']

    # Test local
    local_payload = {"text": text, "model": "ots-mbert"}
    try:
        local_resp = requests.post(local_url, json=local_payload, timeout=10)
        local_data = local_resp.json()
        local_label = local_data.get('label')
        local_prob = local_data.get('probability')
        local_time = local_data.get('processing_time')
    except Exception as e:
        local_label = "error"
        local_prob = 0
        local_time = 0

    # Test external
    external_payload = {"text": text, "model": "bert"}
    try:
        external_resp = requests.post(external_url, json=external_payload, timeout=10)
        external_data = external_resp.json()
        external_label = external_data.get('label')
        external_prob = external_data.get('probability')
        external_time = external_data.get('processing_time')
    except Exception as e:
        external_label = "error"
        external_prob = 0
        external_time = 0

    result = {
        "text": text,
        "expected": expected,
        "category": category,
        "local_label": local_label,
        "local_prob": local_prob,
        "local_time": local_time,
        "external_label": external_label,
        "external_prob": external_prob,
        "external_time": external_time
    }
    results.append(result)

# Save results
with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Testing completed. Results saved to test_results.json")