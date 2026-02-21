Medical AI Project: Classification, Report Generation, and Semantic Retrieval
Overview

This repository presents a three-part medical AI system developed using the PneumoniaMNIST dataset. The project explores discriminative modeling, multimodal report generation, and semantic image retrieval.

The tasks collectively demonstrate:
Supervised medical image classification
Vision-language report generation
Transformer-based semantic image retrieval using vector databases

Repository Structure
medical-ai-project/
│
├── data/                        # Data loading and preprocessing utilities
├── models/                      # Model architectures and saved weights
├── task1_classification/        # CNN classifier
├── task2_report_generation/     # Visual language model
├── task3_retrieval/             # Semantic search system
├── reports/                     # Markdown analysis reports
├── notebooks/                   # Colab notebooks
├── requirements.txt
└── README.md

Task 1: Pneumonia Classification
Objective

Compare a custom CNN with a pretrained ResNet18 for pneumonia detection.
Key Results
SimpleCNN Accuracy: 0.8478
ResNet18 Accuracy: 0.8301
SimpleCNN achieved higher recall (0.9872)

Report
See:

reports/task1_classification_report.md

Task 2: Medical Report Generation

Objective:
Generate radiology-style reports from chest X-rays using a visual language model.

Model Used
LLaVA-1.5-7B (open-source multimodal model)

Key Findings
Structured prompts significantly improved medical relevance.
Model captures pneumonia-related terminology.
Alignment score: 0.3846

Report
See:

reports/task2_report_generation.md

Task 3: Semantic Image Retrieval System

Objective:
Build a content-based image retrieval (CBIR) system using transformer embeddings and FAISS.

Embedding Model
CLIP (ViT-B/32) vision encoder

Vector Database
FAISS IndexFlatL2

Performance
Mean Precision@5: 0.8490

Running the Retrieval System
From task3_retrieval/:
Extract embeddings:

python src/embedding_extractor.py

Build index:

python src/build_index.py

Evaluate:

python src/evaluate.py

Visualize:

python src/visualize.py

Report

See:
reports/task3_retrieval_system.md

Requirements
Install dependencies:

pip install -r requirements.txt


Future Work

Medical-specific embedding models (MedCLIP, BioViL-T)
Explainable retrieval (Grad-CAM)
Advanced retrieval metrics (mAP, Recall@k)
Domain-specific fine-tuning

Author

Owais Ahmad
Medical AI Research Project