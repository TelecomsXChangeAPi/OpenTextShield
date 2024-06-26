# Open Text Shield (OTS) on Apple Silicon

Open Text Shield (OTS) is now fully optimized for Apple Silicon (M1, M2, etc.), bringing unparalleled efficiency and speed to OTS training and inference, By harnessing the advanced capabilities of BERT (Bidirectional Encoder Representations from Transformers), OTS BERT model for Apple Silicon not only elevates the effectiveness of analyzing and processing textual data but also significantly boosts performance. Training models on Apple Silicon is now faster, and inference speeds are improved by at least 50%, ensuring even higher levels of protection and operational efficiency.

![ots-mlx](https://github.com/TelecomsXChangeAPi/OpenTextShield/assets/19316784/d06058ab-bf5b-4136-84e7-82a8c10b07a3)


## Installation

To get started with OTS on apple silicon, you'll need to install the necessary dependencies. Run the following command in your terminal:

```bash
pip3 install -r requirements.txt
```

## Usage

### Loading the BERT Model
Before starting the training process, you need to load the BERT model. Execute the following command to accomplish this:

```bash
python3 load_bert.py
```
## Training the OTS Model

To begin training the Open Text Shield model with MLX Bert, use the command below:

```bash
python train_ots.py
```

Successful output:

```bash
$ python3 train_ots.py 
Detected encoding: UTF-8-SIG
Epoch 1/3 completed.
Epoch 2/3 completed.
Epoch 3/3 completed.
Test Accuracy: 99.72%
```

#### Inference Performance Benchmark on Apple Silicon M1 Pro

| Metric                        | Value                   |
|-------------------------------|-------------------------|
| **Inference Speed**           | 54 SMS messages/second  |
| **Tested Platform**           | Apple Silicon M1 Pro    |





#### Training Process

![6AZuNzub7YUb3aTsnzpsiK](https://github.com/TelecomsXChangeAPi/OpenTextShield/assets/19316784/bbce8f96-b3b3-4beb-9e78-417b47a09e15)


## Contact and Acknowledgements

We appreciate your interest in MLX Bert for OTS and welcome any questions, feedback, or contributions. Please feel free to reach out to us via the following channels:

### For OTS inquiries:
- **LinkedIn**: [Ameed Jamous](https://www.linkedin.com/in/ameedjamous/)
- **Email**: [a.jamous@telecomsxchange.com](mailto:a.jamous@telecomsxchange.com)
- **GitHub**: [TelecomsxchangeAPI/Open-Text-Shield](https://github.com/TelecomsxchangeAPI/Open-Text-Shield)

### For MLX-BERT inquiries:
- **LinkedIn**: [Tim Cvetko](https://www.linkedin.com/in/tim-cvetko-32842a1a6/)
- **Gmail**: [cvetko.tim@gmail.com](mailto:cvetko.tim@gmail.com)
