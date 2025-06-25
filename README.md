Invoice Extractor
Invoice Extractor is a lightweight Streamlit application powered by Google's Gemini Vision API. It allows users to upload an image of an invoice and extract structured information using natural language prompts. This project demonstrates how large multimodal models can be integrated into real-world document understanding tasks such as invoice parsing and data extraction.

Project Overview
This application allows users to:

Upload invoice images (.jpg, .jpeg, .png)

Enter custom text-based prompts to query the invoice (e.g., "What is the total amount?")

Receive structured responses using Gemini Pro Vision (gemini-1.5-flash)

The app is designed for:

Document intelligence applications

AI-based OCR augmentation

Demonstrating generative model capabilities in financial automation

Technologies Used
Streamlit – Interactive front-end web application

Google Generative AI SDK – Access to Gemini models

PIL (Pillow) – Image preprocessing

Python (3.10+)

dotenv – Environment variable management

Folder Structure
bash
Copy
Edit
invoice-extractor/
│
├── app.py                 # Main Streamlit application
├── .env                  # Environment variables (API key)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
Setup Instructions
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/invoice-extractor.git
cd invoice-extractor
Create a Virtual Environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Add Your Google API Key
Create a .env file in the root directory:

ini
Copy
Edit
GOOGLE_API_KEY=your_gemini_api_key_here
Run the Application

bash
Copy
Edit
streamlit run app.py
Example Use Case
Upload an image of an invoice and input a prompt such as:

vbnet
Copy
Edit
What is the total invoice amount?
Who is the sender and recipient?
When was the invoice issued?
The model will analyze the image and return a concise response based on the prompt.

Potential Enhancements
Export extracted data as JSON or CSV

Add PDF support

Integrate OCR fallback (e.g., Tesseract)

Enable multi-language support

Connect to a database for storing invoice records

Notes
The Gemini API key is required to use this app. Sign up via Google AI Studio to obtain one.

Ensure the uploaded invoices are high-quality for best results.

License
This project is released under the MIT License.
