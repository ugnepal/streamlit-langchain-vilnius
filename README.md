# streamlit-langchain-vilnius

🧠 Vilnius RAG Chatbot
This is a Streamlit-based conversational chatbot that uses LangChain and retrieval-augmented generation (RAG) to answer questions exclusively about Vilnius, Lithuania. The model responds only using data sourced from:

Wikipedia article on Vilnius

Official GoVilnius website

A local PDF file (Vilnius_vilnius.pdf)

If a user asks anything unrelated to Vilnius, the app will politely reject the query.

🚀 Features
🔍 Retrieval-Augmented Generation: Combines LLM power with trusted local documents

📄 Multi-source ingestion: Web content + PDF support

🧠 Smart filtering: Strictly answers only Vilnius-specific questions

📚 Cited responses: Shows sources for each answer

🖼️ Interactive UI with Streamlit

🛠️ Installation
Clone the repo

bash
Copy
Edit
git clone https://github.com/your-username/vilnius-rag-chatbot.git
cd vilnius-rag-chatbot
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables

Create a .env file and add your API key:

env
Copy
Edit
SECRET=your-openai-or-proxy-token
📂 File Structure
bash
Copy
Edit
├── Vilnius_vilnius.pdf         # Local PDF source
├── Vilnius.jpg                 # Banner image for Streamlit
├── app.py                      # Main application script
├── .env                        # API keys
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
▶️ Usage
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Ask any question about Vilnius (e.g. tourism, history, facts), and the chatbot will respond only if it can find the answer in one of the predefined sources.

❗ Important Notes
The app uses OpenAI-compatible endpoints for chat and embeddings.

Embeddings are stored in a Chroma vector store.

The RAG pipeline rejects questions not directly related to Vilnius with a fixed response.

🧾 Example Questions
✅ "What is the population of Vilnius?"
✅ "What can I visit in Vilnius?"
❌ "What is the capital of Latvia?" → “Your question is not valid. I can only answer about Vilnius.”

