from transformers import pipeline
import warnings

# Suppress unnecessary system warnings for a cleaner console output
warnings.filterwarnings("ignore")

def run_ner_cpu(text: str, model_name: str = "dslim/bert-base-NER"):
    """
    Perform Named Entity Recognition (NER) using Hugging Face pipeline.
    Forces CPU execution and displays results in a user-friendly table format.
    
    Args:
        text (str): The input text to analyze.
        model_name (str): The pre-trained Hugging Face NER model name.
        
    Returns:
        list: A list of dictionaries containing extracted entities and metadata.
    """
    print("⏳ Loading NER model (Forcing CPU mode)...")
    
    # Initialize the Hugging Face NER pipeline:
    # - device=-1 explicitly forces CPU execution and avoids using GPU (CUDA/MPS)
    # - aggregation_strategy="simple" merges subword tokens (e.g., "##Kong") into whole words ("Hong Kong")
    ner_pipeline = pipeline(
        task="ner",
        model=model_name,
        aggregation_strategy="simple",
        device=-1
    )
    
    # Run inference on the input text
    results = ner_pipeline(text)
    
    # Map raw model entity tags to user-friendly labels
    entity_map = {
        "PER": "👤 Person",
        "ORG": "🏢 Organization",
        "LOC": "📍 Location",
        "MISC": "🏷️ Miscellaneous"
    }
    
    # Print formatted header and original input text
    print("\n" + "=" * 70)
    print("📄 Input Text:")
    print(f"   {text}")
    print("=" * 70)
    
    # Handle edge case where no entities are detected
    if not results:
        print("⚠️ No named entities were detected in the provided text.")
        return []
