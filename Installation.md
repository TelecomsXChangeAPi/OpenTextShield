# Open Text Shield (OTS)

**Open Text Shield (OTS)** is an open-source, real-time filtering and classification tool for SMS and messaging traffic. It helps detect and block spam, phishing, and malicious content, providing a powerful solution for telecom operators, messaging providers, and any application that processes message traffic, such as those using SMPP, SIP, RCS, or SMTP.

## Features
- **Real-Time Detection**: Instantly classify and block spam, phishing, and other unwanted messages.
- **Machine Learning Models**: Leverages advanced models like BERT to ensure high accuracy.
- **Flexible Integration**: Compatible with any application capable of making HTTP requests, including those using SMPP, SIP, RCS, and SMTP.
- **Web API**: Simple and efficient API for message analysis and predictions.
- **Open Source**: Free to use and modify, with commercial support available for advanced use cases.

## Quick Start with Docker

Setting up **Open Text Shield** is quick and easy with Docker. Follow these steps to get started:

### 1. Pull the Latest Docker Image
```bash
docker pull telecomsxchange/opentextshield:latest
```

### 2. Run the Docker Container
```bash
docker run -d -p 8002:8002 telecomsxchange/opentextshield:latest
```

### 3. Send a Message for Prediction
Once the container is running, you can send HTTP requests to the API to classify messages.

Example `curl` request:
```bash
curl -X POST "http://localhost:8002/predict/" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d "{\"text\":\"Your SMS content here\",\"model\":\"bert\"}"
```

### Example Response:
```json
{
  "label": "ham",
  "probability": 0.9971883893013,
  "processing_time": 0.6801116466522217,
  "Model_Name": "OTS_mBERT",
  "Model_Version": "bert-base-uncased",
  "Model_Author": "TelecomsXChange (TCXC)",
  "Last_Training": "2024-03-20"
}
```

## API Documentation

The API accepts the following parameters:

- `text`: The message content to be analyzed.
- `model`: The machine learning model to use (e.g., `bert`).

For more information on API endpoints and parameters, please refer to the full documentation in the [OpenTextShield GitHub Repository](README.md).

## Contributing

We welcome contributions to Open Text Shield! Please check the [contributing guide](CONTRIBUTING.md) for guidelines on how to get involved.

## License

Open Text Shield is open-source and licensed under the [MIT License](LICENSE).
