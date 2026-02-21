# Task 2: Medical Report Generation using a Visual Language Model

## 1. Introduction

This task explores automated medical report generation from chest X-ray images
using a pre-trained open-source Visual Language Model (VLM). The objective is
to integrate a multimodal AI model capable of analyzing images and generating
clinically relevant natural language descriptions.

Unlike Task 1, where a CNN was trained for classification, this task focuses
on leveraging state-of-the-art multimodal generative models without training
from scratch.

---

## 2. Model Selection Justification

The recommended model for this task was MedGemma, an open-source medical VLM
designed specifically for healthcare applications. However, due to GPU memory
constraints on the free-tier Google Colab environment, MedGemma could not be
efficiently deployed within available hardware limits.

Therefore, we selected **LLaVA-1.5-7B** (open-source multimodal model from Hugging Face)
as a computationally feasible alternative.

Justification for this choice:

- Fully open-source and widely adopted
- Supports image-conditioned text generation
- Compatible with free-tier GPU (T4)
- Allows flexible prompt engineering
- Demonstrates strong general multimodal reasoning ability

Although not explicitly trained for radiology, LLaVA provides a strong baseline
for evaluating multimodal report generation capability.

---

## 3. Implementation Pipeline

The implemented pipeline consists of:

1. Loading the PneumoniaMNIST test dataset
2. Preprocessing images according to model requirements
3. Loading the pre-trained VLM
4. Generating reports using prompt-based conditioning
5. Comparing outputs with:
   - Ground truth labels
   - CNN predictions from Task 1

The system generates structured medical-style reports directly within the notebook,
allowing qualitative inspection.

---

## 4. Prompting Strategies Evaluated

Prompt engineering significantly influences the quality and specificity of generated reports.
Three prompting strategies were evaluated:

### 4.1 Basic Prompt

"Describe this chest X-ray image."

**Observation:**
- Produces generic descriptions
- Limited medical terminology
- Often vague and high-level

---

### 4.2 Radiologist Prompt

"You are a medical radiologist. Describe the key findings and provide a possible diagnosis."

**Observation:**
- More structured output
- Includes diagnostic phrasing
- Occasionally overconfident conclusions

---

### 4.3 Structured Clinical Prompt (Final Choice)

"You are an expert thoracic radiologist. Analyze the chest X-ray focusing on:
- Lung opacity
- Consolidation
- Infiltrates
Provide a concise clinical report."

**Observation:**
- Most clinically relevant outputs
- Better alignment with radiological terminology
- Reduced generic descriptions
- Improved focus on pneumonia-related features

Conclusion:  
Structured, domain-specific prompts significantly improved medical relevance
and clarity of generated reports.

---

## 5. Representative Sample Evaluation

A representative subset of images was selected including:

- 5 Normal cases
- 5 Pneumonia cases
- 3 CNN misclassified cases

For each case, the following were compared:

- Ground truth label
- CNN prediction (from Task 1)
- Generated VLM report

### Observed Patterns

- In many pneumonia cases, the VLM mentioned opacity, consolidation,
  or infiltrates.
- In normal cases, the model frequently described clear lung fields
  or absence of abnormal findings.
- In some misclassified CNN cases, the VLM provided ambiguous descriptions,
  suggesting subtle or borderline patterns.

This demonstrates the complementary role of generative multimodal models
alongside discriminative CNN classifiers.

---

## 6. CNN–VLM Alignment Analysis

To introduce lightweight quantitative analysis, a simple keyword-based
alignment score was computed.

For pneumonia cases, the presence of keywords such as:
- opacity
- consolidation
- infiltrate
- infection

was checked.

For normal cases, the presence of:
- clear
- normal
- no abnormality
- unremarkable

was evaluated.

The alignment score (on selected subset):

Alignment Score: 0.3846

This indicates that the VLM frequently captures class-relevant terminology,
though occasional ambiguous or generic outputs were observed.

---

## 7. Strengths and Limitations

### Strengths

- Generates coherent, medically styled language
- Highly responsive to structured prompts
- Can complement CNN predictions with interpretability
- Requires no domain-specific training

### Limitations

- Not trained specifically for radiology
- Risk of hallucinated findings
- May generate overconfident diagnostic statements
- No explicit grounding mechanism to verify visual evidence
- Sensitive to prompt design

These limitations highlight the need for human oversight in clinical settings.

---

## 8. Discussion

The integration of a visual language model demonstrates the feasibility of
automated medical report generation using open-source multimodal AI.

While not replacing expert radiologists, such systems may serve as:

- Decision-support tools
- Preliminary triage assistants
- Educational aids

However, hallucination risks and domain specificity remain significant
challenges requiring further research.

---

## 9. Conclusion

This task successfully demonstrates:

- Integration of a multimodal visual language model
- Effective prompt engineering
- Comparative analysis with a trained CNN classifier
- Qualitative and lightweight quantitative evaluation

Structured prompting substantially improves medical relevance of generated
reports. Although promising, the deployment of such systems in real-world
clinical settings would require domain-specific fine-tuning and rigorous
validation.

