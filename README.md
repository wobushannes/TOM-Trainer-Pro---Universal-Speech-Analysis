# TOM Trainer Pro — Universal Speech Analysis

**Enterprise-Grade LoRA Training & Theory of Mind Analysis Platform**

---

## TL;DR

TOM Trainer Pro is a research platform for **Theory of Mind (ToM) analysis** and **LoRA-based fine-tuning** of Large Language Models. It detects deception, hidden emotions, and communicative strategies in text — from single sentences to full documents. Built for research and security applications.

---

## Overview

TOM Trainer Pro uses **LoRA fine-tuning** to adapt LLMs for psycholinguistic pattern detection: deception likelihood, hidden emotional states, communicative strategies (avoidance, manipulation, persuasion), and micro-expression hypotheses in text. It is a fine-tuning and analysis platform that learns from annotated data.

---

## Key Features

- **LoRA Training** – Fine-tune LLMs with real-time monitoring (loss, GPU usage, convergence)
- **Automated Reports** – Generate HTML training reports with quality metrics
- **Document Analysis** – PDF and text document analysis with chunk-based ToM processing
- **Real-Time Analysis** – Sentence-level analysis with detailed interpretation
- **Data Quality Metrics** – Monitor emotion diversity, domain balance, and annotation consistency

---

## Supported Models

- Qwen2.5 Family: 1.5B, 4B, 7B Instruct
- Mistral AI: Mistral-7B-Instruct-v0.3
- Microsoft: DialoGPT-medium (legacy)

---

## System Requirements

- Python 3.9+ (3.10+ recommended)
- RAM: 16 GB (32 GB recommended)
- GPU: NVIDIA 8 GB+ VRAM (12 GB+ recommended)
- Storage: 20 GB+ (50 GB+ recommended)

---

## Training Data Format

## Field Descriptions

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `input` | String | - | The text to analyze |
| `tom.deception_likelihood` | Float | 0.0–1.0 | Probability of deception |
| `tom.detected_true_emotions` | Array | - | Detected emotions |
| `tom.hidden_intent_candidate` | String | - | Suspected intent |
| `emotion_dynamics.emotional_volatility` | Float | 0.0–1.0 | Emotional volatility score |
| `emotion_dynamics.micro_expression_hypothesis` | String | - | Hypothesis about micro-expressions |
| `communication_style.communication_style` | String | - | Communication style |
| `communication_style.persuasion_strategy` | String | - | Persuasion strategy |

---

## Quick Start

A demo dataset with 10 entries is included (`demo_training_data.json`).

1. Start the GUI: `python tom_trainer_pro_gui.py`
2. Select a model (e.g., `Qwen/Qwen2.5-1.5B-Instruct`)
3. Load the demo dataset (or your own)
4. Convert raw data to training format
5. Start training (3–5 epochs recommended)
6. Load the trained LoRA for analysis

---

## Important Note on Training Data

I do **not** provide support for generating training data or annotating ToM features. High-quality training data requires deep psycholinguistic expertise, understanding of Theory of Mind concepts, experience with emotion annotation, and knowledge of communication strategies.

---

## Use Cases

1. **Political Speech Analysis** – Detect deception, emotional manipulation, and hidden intent in public discourse.
2. **Forensic Linguistics** – Analyze witness statements, interrogation protocols, and credibility assessments.
3. **Psychological Research** – Study emotion dynamics in therapy sessions, couple communication, and child development.
4. **Security Analysis** – Early detection of manipulation attempts, whistleblower analysis, risk assessment in negotiations.

---

## Professional Collaboration

I have trained multiple LoRAs with this format and offer the following services for a fee:

| Service | Description |
|---------|-------------|
| **Data Annotation** | Professional ToM annotation for your domain |
| **LoRA Training** | Custom LoRAs for specific use cases |
| **Quality Assurance** | Data quality audits and optimization |
| **Deployment** | Integration into existing systems |

---

## Contact (Application Process)

Before contacting me, please send the following information:

- **Project Type:** [Research / Company / Startup / Other]
- **Application Area:** [e.g., Political Analysis / Market Research]
- **Data Volume:** [Number of sentences / documents]
- **Timeline:** [Deadline / project duration]
- **Budget:** [Range / indication]
- **Prior Knowledge:** [Experience with NLP / ToM / LoRA]
- **Expectations:** [What should the system deliver?]

**Email:** `blende_32@protonmail.com`
**Subject:** `[ToM Project] - [Your Organization]`
**Response Time:** 48–72 hours (with complete information)

I only respond to inquiries that contain **all** of the above points. Incomplete requests will not be processed.

---

## Ethical Considerations

TOM Trainer Pro is a **high-risk AI research tool**. Textual analysis — especially deception detection and intent analysis — carries significant risks of misuse in surveillance, social control, and discrimination.

**This project is guided by the same ethical principles as MouthMind:**

1. **Transparency Without Vulnerability** – The architecture and approach are documented. Code, weights, and training data are not public.
2. **Privacy by Design** – The system processes text, not personal identifiers. No data is stored or shared without explicit consent.
3. **Controlled Access** – Access is restricted to institutions with proven research experience and a clear ethical framework. Commercial use is strictly prohibited.
4. **Active Misuse Prevention** – Use is monitored. In case of misuse, access is revoked.
5. **Commitment to Public Benefit** – Findings and insights are shared. The tool itself remains controlled.

---

## Disclaimer

This tool is designed for **research purposes only**. Analysis results are **probabilistic** and **not suitable as sole decision-making criteria**. Misinterpretation is possible. Use is at your own risk.

---

## License

This project is **not open source**. All rights reserved. © 2026 Johannes Wobus — TOM Trainer Pro Research

---

## FAQ

**Can I get the code or model?** No. Code, weights, and training data are not public — and never will be.

**I'm a researcher / student. Can I get access?** No. Access is restricted to institutions with proven research experience and a clear ethical framework.

**Can I use it commercially?** No. Commercial use is strictly prohibited. No licenses. No exceptions.

**Why so restrictive?** Because deception detection and intent analysis have misuse potential. We take this seriously.

**Can I collaborate?** Yes — if you're from a university, institute, or comparable research environment. Contact us with your profile, institution, and concrete proposal. All other inquiries will be ignored.

**Can I discuss the research with you?** Yes — in academic or policy contexts. Contact us with a clear proposal.

**Can I hire you as a consultant?** We offer consulting and research collaboration for institutions with a clear ethical framework. Contact us for details.

---

## Contributions

Contributions are welcome — but only under the same ethical framework.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m 'Add some AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## Contact

Serious inquiries only.

**ProtonMail:** `blende_32@protonmail.com`
**Threema:** `BA46EWMP`

**Before contacting us:**

- Provide full name and institution
- State your concrete purpose
- Don't ask for code or access — it will be ignored

---

*"We show what's possible — not how it's done."*
