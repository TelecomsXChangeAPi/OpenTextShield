# OpenTextShield Feedback API

The OpenTextShield API includes feedback endpoints to collect user feedback on classification results.

## Endpoints

### POST /feedback/
Submit feedback about classification results.

**Request Body:**
```json
{
  "content": "Original text that was classified",
  "feedback": "User feedback about the classification",
  "thumbs_up": true,
  "thumbs_down": false,
  "user_id": "optional_user_identifier",
  "model": "ots-mbert"
}
```

**Response:**
```json
{
  "message": "Feedback received successfully",
  "feedback_id": "uuid-feedback-id"
}
```

### GET /feedback/download/{model_name}
Download feedback CSV file for a specific model.

**Parameters:**
- `model_name`: Model name (e.g., "ots-mbert")

**Response:**
- CSV file with feedback data
- Headers: FeedbackID, Timestamp, UserID, Content, Feedback, ThumbsUp, ThumbsDown, Model

## Example Usage

### Submit Feedback with cURL
```bash
curl -X POST "http://localhost:8002/feedback/" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "URGENT! Click this link now!",
    "feedback": "Correctly identified as spam",
    "thumbs_up": true,
    "thumbs_down": false,
    "user_id": "user123",
    "model": "ots-mbert"
  }'
```

### Download Feedback File
```bash
curl "http://localhost:8002/feedback/download/ots-mbert" -o feedback.csv
```

## Testing

Run the feedback API tests:
```bash
# Python test script
python3 test_feedback_api.py

# cURL test script
./test_feedback_curl.sh
```

## Data Storage

Feedback is stored in CSV files in the `feedback/` directory:
- File format: `feedback_{model_name}.csv`
- Example: `feedback/feedback_ots-mbert.csv`

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Invalid model name or bad request
- `404`: Feedback file not found
- `500`: Internal server error