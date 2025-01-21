# Document Summarizer

A web-based application that provides powerful summarization and key point extraction for uploaded documents and images using state-of-the-art machine learning models. The app supports both PDF and image files (JPEG, PNG) and uses OCR for text extraction from images.

## Features

- Upload and process PDF and image files.
- Extract text content from PDFs and images (using OCR).
- Summarize extracted text into concise, meaningful summaries.
- Identify and display key points from the uploaded documents.
- Interactive web interface built using **Streamlit**.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**:
  - LangChain
  - Sentence Transformers
  - PyTorch
  - Hugging Face Transformers
  - PyPDFLoader
  - Tesseract OCR
  - FastAPI
  - Uvicorn

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/document-summarizer.git
   cd document-summarizer
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the project directory.
   - Add your Hugging Face token:
     ```
     HUGGING_FACE_TOKEN=your_hugging_face_api_token
     ```

5. **Install Tesseract OCR** (if not already installed):
   - For Linux (Ubuntu):
     ```bash
     sudo apt install tesseract-ocr
     ```
   - For Mac (using Homebrew):
     ```bash
     brew install tesseract
     ```
   - For Windows:
     Download and install Tesseract from [here](https://github.com/tesseract-ocr/tesseract).

## Usage

1. **Run the app**:
   ```bash
   streamlit run app.py
   ```

2. **Upload a file**:
   - Supported formats: PDF, JPG, PNG.
   - Drag and drop the file or select it using the upload button.

3. **Summarize**:
   - Click the "Summarize" button to process the file.
   - View the extracted text, summary, and key points in the app interface.

## Project Structure

```
document-summarizer/
├── app.py               # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── README.md            # Project documentation
├── data/                # Directory for uploaded files (auto-created)
└── ...
```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face for their pre-trained models.
- LangChain for document processing.
- Streamlit for the interactive web interface.
- Tesseract OCR for image-to-text extraction.

--- 

You can save this content in a file named `README.md`. Let me know if you'd like to customize or add any additional sections.
