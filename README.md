# 🩺 Medical AI Project: Classification, Report Generation, and Semantic Retrieval

## 📌 Overview

This repository presents a three-part **medical AI system** developed using the PneumoniaMNIST dataset.  
The project explores discriminative modeling, multimodal report generation, and semantic image retrieval.

The system demonstrates:

- Supervised medical image classification  
- Vision-language report generation  
- Transformer-based semantic image retrieval using vector databases  

This project reflects an end-to-end multimodal AI pipeline for medical imaging applications.

---

## 🗂 Repository Structure

# 🩺 Medical AI Project: Classification, Report Generation, and Semantic Retrieval

## 📌 Overview

This repository presents a three-part **medical AI system** developed using the PneumoniaMNIST dataset.  
The project explores discriminative modeling, multimodal report generation, and semantic image retrieval.

The system demonstrates:

- Supervised medical image classification  
- Vision-language report generation  
- Transformer-based semantic image retrieval using vector databases  

This project reflects an end-to-end multimodal AI pipeline for medical imaging applications.

---

## 🗂 Repository Structure

medical-ai-project/
│
├── data/ # Data loading and preprocessing utilities
├── models/ # Model architectures and saved weights
├── task1_classification/ # CNN classifier implementation
├── task2_report_generation/ # Vision-Language model implementation
├── task3_retrieval/ # Semantic image retrieval system
├── reports/ # Markdown analysis reports
├── notebooks/ # Colab notebook(s)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

# 🧠 Task 1: Pneumonia Classification

## 🎯 Objective

Compare a custom CNN with a pretrained ResNet18 model for pneumonia detection using chest X-ray images.

## 📊 Key Results

- **SimpleCNN Accuracy:** 0.8478  
- **ResNet18 Accuracy:** 0.8301  
- **SimpleCNN Recall:** 0.9872  

The custom CNN achieved higher recall, making it particularly effective in identifying pneumonia-positive cases.

## 📄 Detailed Report


---

# 🧠 Task 1: Pneumonia Classification

## 🎯 Objective

Compare a custom CNN with a pretrained ResNet18 model for pneumonia detection using chest X-ray images.

## 📊 Key Results

- **SimpleCNN Accuracy:** 0.8478  
- **ResNet18 Accuracy:** 0.8301  
- **SimpleCNN Recall:** 0.9872  

The custom CNN achieved higher recall, making it particularly effective in identifying pneumonia-positive cases.

## 📄 Detailed Report

reports/task1_classification_report.md


---

# 📝 Task 2: Medical Report Generation

## 🎯 Objective

Generate radiology-style diagnostic reports from chest X-ray images using a multimodal vision-language model.

## 🤖 Model Used

- **LLaVA-1.5-7B** (Open-source Vision-Language Model)

## 🔍 Key Findings

- Structured prompting significantly improved clinical relevance  
- Generated reports included pneumonia-related terminology  
- **Alignment Score:** 0.3846  

## 📄 Detailed Report


---

# 📝 Task 2: Medical Report Generation

## 🎯 Objective

Generate radiology-style diagnostic reports from chest X-ray images using a multimodal vision-language model.

## 🤖 Model Used

- **LLaVA-1.5-7B** (Open-source Vision-Language Model)

## 🔍 Key Findings

- Structured prompting significantly improved clinical relevance  
- Generated reports included pneumonia-related terminology  
- **Alignment Score:** 0.3846  

## 📄 Detailed Report

reports/task2_report_generation.md


---

# 🔎 Task 3: Semantic Image Retrieval System

## 🎯 Objective

Build a Content-Based Image Retrieval (CBIR) system using transformer-based image embeddings and FAISS vector search.

## 🧩 Embedding Model

- **CLIP (ViT-B/32)** vision encoder

## 🗄 Vector Database

- **FAISS (IndexFlatL2)** for efficient similarity search

## 📈 Performance

- **Mean Precision@5:** 0.8490  

The system effectively groups semantically similar pneumonia cases based on learned visual representations.

---

## ▶ Running the Retrieval System

Navigate to:


---

# 🔎 Task 3: Semantic Image Retrieval System

## 🎯 Objective

Build a Content-Based Image Retrieval (CBIR) system using transformer-based image embeddings and FAISS vector search.

## 🧩 Embedding Model

- **CLIP (ViT-B/32)** vision encoder

## 🗄 Vector Database

- **FAISS (IndexFlatL2)** for efficient similarity search

## 📈 Performance

- **Mean Precision@5:** 0.8490  

The system effectively groups semantically similar pneumonia cases based on learned visual representations.

---

## ▶ Running the Retrieval System

Navigate to:

task_semantic_retreival/

### 1️⃣ Extract embeddings

```bash
python src/embedding_extractor.py

### 2️⃣ Build FAISS index
python src/build_index.py

### 3️⃣ Evaluate retrieval (Precision@k)
python src/evaluate.py

### 4️⃣ Visualize retrieval results
reports/task3_retrieval_system.md

### ⚙ Installation

Clone the repository:
git clone https://github.com/yourusername/medical-ai-project.git
cd medical-ai-project

Install dependencies:
pip install -r requirements.txt

#### 🔬 Future Improvements

Integration of medical-specific embedding models (MedCLIP, BioViL-T)
Explainable retrieval using Grad-CAM
Advanced retrieval metrics (mAP, Recall@k)
Domain-specific fine-tuning on larger radiology datasets

#### 👨‍💻 Author

Owais Bhat
AI/ML Researcher | Medical Imaging & Multimodal AI

#### ⭐ Project Highlights

End-to-end medical AI pipeline
CNN vs pretrained model comparison
Vision-language medical report generation
Transformer-based semantic retrieval with FAISS
Quantitative and qualitative evaluation

If you find this project useful, feel free to ⭐ the repository.

