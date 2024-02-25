## What is Open Text Shield (OTS) API interface?

OTS (Open Source Text Shield) is an AI-driven solution designed to enhance the security of telecom networks by detecting and filtering spam and phishing messages in real time. This application leverages both BERT and FastText models for efficient text classification.

The OTS API serves as an intermediary, facilitating communication between the client and the BERT and FastText models. For a clearer insight into how this process works, refer to the diagram provided below. To begin interacting with the models via HTTP, your initial step should be to initiate the API server.

### Flow

![7aa7Q3P3uds5KEtfTww76j](https://github.com/TelecomsXChangeAPi/OpenTextShield/assets/19316784/c2fbf6a4-0fff-45b5-b8d3-cbbc4312b7df)



## Getting Started

### Prerequisites

- Python 3.8 or later
- FastAPI
- pydantic
- torch
- transformers
- fasttext

You can install the necessary libraries using pip:

```bash
pip install fastapi pydantic torch transformers fasttext
```

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/TelecomsXChangeAPi/OpenTextShield/

```

Navigate to the cloned directory:

```bash
cd OpenTextShield
```

### Running the Application

Start the server by running:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

The application will be available at `http://localhost:8001`.

### Usage

#### Predicting SMS

To predict if an SMS is spam, phishing, or ham (regular message), send a POST request to `/predict/` with a JSON body containing the SMS text and the model to use (`bert` or `fasttext`).

Example using curl:

```bash
curl -X POST "http://localhost:8001/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"text\":\"Your SMS content here\",\"model\":\"bert\"}"
```

#### Feedback Loop

To provide feedback on predictions, send a POST request to `/feedback-loop/` with relevant feedback data.

Example using curl:

```bash
curl -X POST "http://localhost:8001/feedback-loop/" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"content\":\"SMS content\",\"feedback\":\"Your feedback here\",\"thumbs_up\":true,\"thumbs_down\":false,\"user_id\":\"user123\",\"model\":\"bert\"}"
```

#### Download Feedback

To download the feedback data for a specific model, send a GET request to `/download-feedback/{model_name}`.

Example using curl:

```bash
curl -X GET "http://localhost:8001/download-feedback/bert"
```


## Acknowledgements

Special thanks to the team at TelecomsXChange (TCXC) for their invaluable contributions to this project.

