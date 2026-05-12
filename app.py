import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model from HuggingFace
MODEL_NAME = "itz-dhanu/NeoPharmabert"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("✅ Model loaded!")

LABELS = ["Patient", "Reporter", "Drug", "Event"]

def classify_icsr(abstract):
    if not abstract.strip():
        return "Please paste an abstract first.", "", "", "", ""

    inputs = tokenizer(
        abstract,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()

    preds = [p > 0.5 for p in probs]

    patient  = "✅ Yes" if preds[0] else "❌ No"
    reporter = "✅ Yes" if preds[1] else "❌ No"
    drug     = "✅ Yes" if preds[2] else "❌ No"
    event    = "✅ Yes" if preds[3] else "❌ No"

    all_valid = all(preds)
    verdict = "✅ VALID ICSR" if all_valid else "❌ INVALID ICSR"

    confidence = (
        f"Patient: {probs[0]:.2f} | "
        f"Reporter: {probs[1]:.2f} | "
        f"Drug: {probs[2]:.2f} | "
        f"Event: {probs[3]:.2f}"
    )

    return verdict, patient, reporter, drug, event, confidence


# Examples
examples = [
    ["A 45-year-old female developed anaphylaxis after receiving penicillin injection, as reported by her physician. The drug was immediately stopped and the patient recovered fully."],
    ["Hepatotoxicity has been associated with isoniazid in several published literature reviews and meta-analyses."],
    ["We report a case of a 67-year-old male who developed acute renal failure following ibuprofen use, as noted by his nephrologist. The drug was discontinued and renal function recovered within 2 weeks."],
]

# UI
with gr.Blocks(title="NeoPharmabert") as demo:

    gr.Markdown("""
    # 🧬 NeoPharmabert — ICSR Validity Classifier
    **Fine-tuned PubMedBERT** for automated pharmacovigilance literature screening.
    
    Paste a PubMed abstract to classify it across **4 ICSR validity criteria**.
    
    | Criterion | Description |
    |-----------|-------------|
    | Patient | Is an identifiable patient mentioned? |
    | Reporter | Is an identifiable reporter mentioned? |
    | Drug | Is a suspect drug mentioned? |
    | Event | Is an adverse event mentioned? |
    
    > **F1 Macro: 0.976** · Trained on ~5,000 labeled abstracts · Base: PubMedBERT
    """)

    with gr.Row():
        with gr.Column():
            abstract_input = gr.Textbox(
                lines=8,
                label="PubMed Abstract",
                placeholder="Paste your PubMed abstract here..."
            )
            classify_btn = gr.Button("🔍 Classify ICSR", variant="primary")

        with gr.Column():
            verdict_output   = gr.Textbox(label="Overall Verdict")
            patient_output   = gr.Textbox(label="Patient")
            reporter_output  = gr.Textbox(label="Reporter")
            drug_output      = gr.Textbox(label="Drug")
            event_output     = gr.Textbox(label="Event")
            confidence_output = gr.Textbox(label="Confidence Scores")

    classify_btn.click(
        fn=classify_icsr,
        inputs=abstract_input,
        outputs=[
            verdict_output,
            patient_output,
            reporter_output,
            drug_output,
            event_output,
            confidence_output
        ]
    )

    gr.Examples(
        examples=examples,
        inputs=abstract_input
    )

    gr.Markdown("""
    ---
    Built by **Dhanusree** · 
    [GitHub](https://github.com/itz-dhanu/NeoPharmBert) · 
    [Model](https://huggingface.co/itz-dhanu/NeoPharmabert)
    """)

demo.launch()
