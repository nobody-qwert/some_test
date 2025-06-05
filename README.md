# Risk Assessment App - Quick Setup

## Setup Steps

1. **Python Environment**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **LMStudio Configuration**
   - Install [LMStudio](https://lmstudio.ai/)
   - Open LMStudio → Developer tab → Load model
   - Copy API endpoint URL (e.g., `http://localhost:1234`)
   - Edit `.env` file:
     ```
     LLM_PROVIDER=lmstudio
     LLM_MODEL=model
     LMSTUDIO_BASE_URL=http://your-lmstudio-endpoint
     ```

3. **Run Application**
   ```bash
   python src/main.py --topic "artificial intelligence" --target "investment risks in AI" --days 7 --articles 5 --detail medium
   ```

## Required Arguments
- `--topic`: News search topic
- `--target`: Target for relevance analysis
- `--days`: Days to look back (default: 7)
- `--articles`: Articles to analyze (default: 10)
- `--detail`: Detail level (low/medium/high)
