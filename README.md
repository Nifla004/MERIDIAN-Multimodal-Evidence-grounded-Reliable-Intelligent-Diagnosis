# MERIDIAN-Multimodal-Evidence-grounded-Reliable-Intelligent-Diagnosis with Integrated Abstention Networks
An independent research extension of multimodal medical AI — investigating how a system that jointly reasons over CT imaging, radiology reports, and structured clinical data can become more trustworthy by explicitly modeling cross-modal evidence consistency, uncertainty, distribution shift, and the possibility of abstaining when reliable prediction is not justified.

## **Relationship to Merlin**
This project is inspired by and designed as an explicit research extension of the Merlin foundation model from Prof. Curt Langlotz's group at Stanford. Merlin investigates large-scale multimodal 3D medical image pretraining and demonstrates that jointly encoding CT volumes with radiology reports produces powerful transferable representations.

MERIDIAN asks a different and complementary research question: given such a multimodal system, can we build a reliability layer that tells a clinician not just what the system predicts, but whether that prediction should be trusted? MERIDIAN does not reproduce Merlin, does not claim to replace Merlin, and explicitly credits the original Merlin contributions. Every design decision in MERIDIAN is an independent research choice. The comparison is always framed as building on top of the direction Merlin opened, not competing with it.

## **Central Research Question**
Can a multimodal medical AI system that jointly reasons over medical images and clinical language become more trustworthy by explicitly modeling cross-modal evidence consistency, uncertainty, distribution shift, and the possibility of abstaining when reliable prediction is not justified?

The purpose is not a marginally higher AUC. The purpose is a system a radiologist can actually trust.

## **Architecture**
````markdown

INPUT LAYER
  │
  ├── CT Volume → 3D Med-Image Encoder → img_emb (256)
  │
  ├── Radiology Report → Clinical Text Encoder → txt_emb (256)
  │
  └── Structured EHR → EHR Encoder → ehr_emb (128)
                          │
                          ▼
              Gated Cross-Modal Attention
              ┌──────────────────────────┐
              │  Image queries Text      │ → gate_img, gate_txt
              │  Text queries Image      │ → cross-modal attention
              │  EHR independently fused │ → gate_ehr
              └──────────────────────────┘
                          │
                          ▼
                       fused (512)
                          │
         ┌────────────────┼──────────────────┐
         ▼                ▼                  ▼
   Prediction Head  Uncertainty Head    Binary Head
   (5 classes)      (log variance)       (abnormal?)
         │                │                  │
         └────────────────┼──────────────────┘
                          ▼
             RELIABILITY LAYER
              (Independent Contributions)
                          │
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  ├── Cross-Modal Evidence Alignment (CMEA)              │
  │  ├── MC-Dropout Uncertainty (Aleatoric + Epistemic)     │
  │  ├── Conformal Prediction (90% coverage guarantee)     │
  │  ├── Distribution Shift Detection (MMD + Energy)       │
  │  ├── Missing Modality Awareness                         │
  │  ├── Selective Prediction / Abstention Engine           │
  │  └── Structured Counterfactual Evaluation               │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
                          │
                          ▼
              Clinical Decision:
              PREDICT / ESCALATE / ABSTAIN
````
## **The Eight Independent Contributions**
**Contribution 1 — Cross-Modal Evidence Alignment (CMEA).** The system evaluates whether the visual evidence in the CT representation is consistent with the information encoded in the radiology report. It computes cosine similarity between image and text embeddings and uses this score as an additional runtime reliability signal. The current prototype investigates whether cross-modal consistency can provide useful information about prediction reliability, with further validation required on real clinical datasets.

**Contribution 2 — Aleatoric and Epistemic Uncertainty Decomposition.** Using MC-Dropout with thirty forward passes, MERIDIAN decomposes predictive uncertainty into aleatoric uncertainty (irreducible, from ambiguous cases) and epistemic uncertainty (reducible, from model ignorance). A clinician receiving a high-epistemic-uncertainty case knows more training data would help. A high-aleatoric case is genuinely ambiguous and needs expert review regardless.

**Contribution 3 — Conformal Prediction Coverage Guarantee.** MERIDIAN wraps its classifier in a conformal predictor calibrated on a held-out set. This provides a mathematically guaranteed coverage statement: at ninety percent confidence, the true diagnosis is contained in the prediction set with probability at least ninety percent. When the system is uncertain, it returns multiple plausible diagnoses rather than forcing a single wrong answer.

**Contribution 4 — Distribution Shift Detection.** Using Maximum Mean Discrepancy and Energy Distance computed in the fused representation space, MERIDIAN quantifies how far a new case is from the training distribution. Institution-specific shift is measured and correlated with AUC degradation, providing a runtime signal that the model may be operating out of distribution.

**Contribution 5 — Missing Modality Awareness with Gate Adaptation.** Real clinical settings always involve missing data. MERIDIAN's gated attention architecture learns to suppress gates for unavailable modalities and redistribute information across available ones. The system explicitly reports its current modality configuration and adjusts uncertainty accordingly — no silent imputation, no crashes.

**Contribution 6 — Selective Prediction and Abstention Engine.** MERIDIAN combines four reliability signals — predictive entropy, CMEA score, gate availability, and conformal set size — into a single Reliability Score. Based on this score each case is classified as PREDICT (high confidence), ESCALATE (borderline, needs expert), or ABSTAIN (insufficient evidence). The risk-coverage curve proves that abstaining on low-reliability cases reduces error rate on the remaining predictions.

**Contribution 7 — Structured Counterfactual Evaluation.** Removing disease-specific keywords from the report, zeroing CT features, and normalising EHR values to normal ranges reveals whether the model's prediction changes for scientifically appropriate reasons. A robust model should change prediction when causal evidence is removed and should not change when irrelevant information changes.

**Contribution 8 — Clinical Reliability Dashboard.** Every patient receives a complete reliability report including the prediction, the prediction set, the reliability score, the clinical decision recommendation, the uncertainty decomposition, the CMEA score, and the modality gate values. The dashboard integrates all signals into a single actionable output for the radiologist.

## **Results Summary**
MERIDIAN demonstrates that the reliability layer provides statistically meaningful improvements in trustworthiness beyond point prediction accuracy.

The CMEA score correlates with prediction correctness — cases where image and report are consistent are predicted more accurately, validating that cross-modal alignment is a genuine reliability signal. MC-Dropout uncertainty successfully discriminates correct from incorrect predictions with a large Cohen's d effect size. Conformal prediction achieves the target ninety percent coverage on the test set, meaning the mathematical guarantee is empirically valid. Distribution shift is detected by MMD and correlates with AUC degradation across institutions. The selective prediction engine produces a risk-coverage curve with monotonically decreasing risk as coverage decreases — the system correctly identifies which cases to abstain on. Structured counterfactuals show that removing disease-specific report keywords causes meaningful prediction shifts, suggesting the text encoder is learning clinically relevant patterns rather than spurious correlations.

## **Honest Limitations**
MERIDIAN is a research prototype. The CT features are simulated vectors, not real 3D volumes. Real implementation requires Med3D, SwinUNETR, or similar architectures on NLST or LIDC data. The text encoder is DistilBERT, not a radiology-specific model such as RadBERT or ClinicalBERT. The reliability guarantees are valid under the statistical assumptions of conformal prediction and would need prospective clinical validation before any deployment consideration. MERIDIAN makes no clinical readiness claims.

## **Cell-by-Cell Summary**

The notebook contains twenty-six cells covering dataset generation, all three unimodal encoders, the gated cross-modal attention fusion module, training with multi-task loss, evaluation against three unimodal baselines, all eight independent reliability contributions implemented and evaluated, ablation study, statistical significance testing, honest limitations, complete final report, and zip download.

## **About the Author**
Nifla Nalakath |
BTech in Computer Science and Engineering |
APJ Abdul Kalam Technological University, Kerala, India |
niflanalakath@gmail.com

*MERIDIAN does not claim to reproduce Merlin or achieve state-of-the-art CT diagnosis. It investigates a single scientific question: can explicit reliability modeling make a multimodal medical AI system more trustworthy? The answer, within the constraints of this prototype, is yes — and the mechanisms are measurable, statistically validated, and clinically interpretable.*

