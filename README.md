# Urdu to Roman Transliteration System

A comprehensive deep learning system for translating Urdu text to Roman Urdu using BiLSTM encoder-decoder architecture and custom Byte Pair Encoding (BPE).

## 🚀 Features

- BiLSTM Encoder: 2-layer bidirectional LSTM for robust Urdu text encoding
- LSTM Decoder: 4-layer LSTM for accurate generation
- Custom BPE: Specialized tokenization for both Urdu and Roman scripts
- Web Interface: Interactive Streamlit app with real-time translation
- Batch Processing: Translate multiple sentences simultaneously
- Comprehensive Evaluation: BLEU score, Character Error Rate, and perplexity metrics

## 📊 Model Performance

### Architecture Details

- Total Parameters: ~30.2 Million
- Encoder: BiLSTM (512 hidden units × 2 layers, bidirectional)
- Decoder: LSTM (512 hidden units × 4 layers)
- Embedding Dimension: 256
- Vocabulary Size: 4,000 BPE tokens each (Urdu/Roman)

### Training Results

- Final Validation Loss: 0.56
- Perplexity: 1.75
- Training Duration: 10 epochs
- Dataset: 21,003 Urdu-Roman sentence pairs from Rekhta Ghazals

## 🎯 Example Translations

| Urdu Input                                | Expected Output                           | Model Output                            | Quality   |
| ----------------------------------------- | ----------------------------------------- | --------------------------------------- | --------- |
| تو کبھی خود کو بھی دیکھے گا تو ڈر جائے گا | tu kabhi khud ko bhi dekhega to dar jaega | tu kabhi khud ko bhi dekhe ega to jaega | Good      |
| میں آپ سے محبت کرتا ہوں                   | main aap se mohabbat karta hun            | main aap se mohabbat karta huun         | Good      |
| آج موسم بہت اچھا ہے                       | aaj mausam bohat acha hai                 | aaj mausam bahut achchha hai            | Excellent |

## 🛠️ Installation & Setup

### Local Installation

```bash
# Clone the repository
git clone https://github.com/hasnaatmalik/urdu2romanUrdu
cd urdu2romanUrdu

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Launch the app
streamlit run app.py
```

## 📁 File Structure

```
├── 📓 main.ipynb              # Complete training pipeline notebook
├── 🌐 app.py                  # Streamlit web interface
├── 🔧 setup_kaggle.py         # One-click Kaggle deployment (optional)
├── 📋 requirements.txt        # Python dependencies
├── 📖 README.md              # This documentation
├── 🚫 .gitignore             # Git ignore rules
├── ⚖️ LICENSE                # MIT license
└── 📊 models/                # Model artifacts (not in git)
    ├── bpe_model.pkl         # Trained BPE tokenizer (~8MB)
    ├── best_model.pt         # Best model checkpoint (~120MB)
    ├── model_info.pkl        # Training metadata
    └── training_curves.png   # Loss visualization
```

## 🖥️ Web Interface Features

### Single Translation

- Input Urdu text and get instant Roman translation
- Shows inference time and token IDs (debug mode)

### Batch Processing

- Process multiple sentences at once
- Line-by-line input format
- Bulk translation results

### Model Evaluation

- Quick evaluation on test dataset
- BLEU score calculation
- Side-by-side comparison (Reference vs Prediction)

### Training Insights

- Loss curves visualization
- Training history from checkpoints
- Performance metrics over time

## 🔧 Technical Implementation

### Data Preprocessing

```python
def normalize_urdu(text):
    # Unicode normalization (NFC)
    # Remove presentation form characters
    # Handle punctuation and whitespace
    # Add sentence boundaries
```

### BPE Tokenization

- Custom implementation for Urdu and Roman scripts
- Character-level fallback for unknown words
- Subword tokenization for better coverage
- Special tokens: `<PAD>`, `<UNK>`, `< SOS >`, `<EOS>`

### Training Configuration

```python
# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 50
TEACHER_FORCING_RATIO = 0.7
DROPOUT = 0.3
WEIGHT_DECAY = 1e-5
```

## 📈 Training Pipeline

0. Data Collection: Urdu Ghazals from Rekhta dataset
1. Preprocessing: Text normalization and cleaning
2. BPE Training: Custom tokenizer for both languages
3. Model Training: Seq2Seq with teacher forcing
4. Validation: BLEU, CER, and perplexity evaluation
5. Checkpointing: Save best model based on validation loss

## 🎨 Usage Examples

### Programmatic Usage

```python
from app import UrduRomanBPE, Seq2SeqModel, load_model, load_bpe

# Load trained components
bpe = load_bpe('bpe_model.pkl')
model, history = load_model('best_model.pt', bpe)

# Translate text
urdu_text = "تو کبھی خود کو بھی دیکھے گا"
# ... (see app.py for complete pipeline)
```

### Web Interface

0. Open the Streamlit app
1. Enter Urdu text in the input box
2. Click "Translate" to get Roman output
3. Explore attention visualization
4. Try batch processing for multiple sentences

## ⚠️ Limitations & Known Issues

### Current Limitations

0. Domain Specificity: Trained primarily on poetry/ghazal text
1. Repetition Issues: Some translations may have repeated tokens
2. Vocabulary Constraints: Limited to 4K BPE tokens per language
3. Decoding Strategy: Uses greedy decoding (no beam search)
4. Computational Requirements: Requires ~4GB RAM for inference

## 📊 Dataset Information

### Statistics

- Total Pairs: 21,003
- Training Set: 16,802 pairs (80%)
- Validation Set: 4,201 pairs (20%)
- Average Length: ~8-12 words per sentence
- Domain: Classical Urdu poetry (Ghazals)

### Preprocessing Steps

0. Unicode normalization (NFC for Urdu, NFKD for Roman)
1. Punctuation standardization
2. Whitespace normalization
3. Sentence boundary detection
4. Quality filtering (length, character set validation)

## 🔬 Evaluation Metrics

### BLEU Score

- Measures translation quality against reference
- Uses smoothing for short sentences
- Calculated at sentence level

### Character Error Rate (CER)

- Edit distance between prediction and reference
- Normalized by reference length
- Lower values indicate better performance

### Perplexity

- Exponential of validation loss
- Measures model uncertainty
- Lower values indicate better language modeling

## 🤝 Contributing

We welcome contributions! Please follow these steps:

0. Fork the repository
1. Create a feature branch (`git checkout -b feature/improvement`)
2. Commit your changes (`git commit -am 'Add improvement'`)
3. Push to the branch (`git push origin feature/improvement`)
4. Create a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/hasnaatmalik/urdu2romanUrdu

# Install development dependencies
pip install -r requirements.txt
```

### Third-Party Licenses

- PyTorch: BSD License
- Streamlit: Apache License 2.0

## 📞 Support & Contact

Made with ❤️ for the Urdu NLP community

Last updated: 02:37 AM PKT on Sunday, September 28, 2025
