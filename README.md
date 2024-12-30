# Query Expansion: Advanced RAG Techniques

This project demonstrates the implementation of the **Query Expansion** technique, an advanced Retrieval-Augmented Generation (RAG) method. By utilizing **ChromaDB’s embedding functions, OpenAI’s API**, and **UMAP**, this technique enhances the performance of a Large Language Model (LLM) by expanding the initial query with related sub-queries and improving retrieval precision from PDF documents.

### Workflow Overview:
1. **Extract and Process Text**: Extract and process raw text from PDF documents into manageable formatting.
2. **Chunk and Embed**: Split the text into manageable chunks and generate vector embeddings using ChromaDB.
3. **Generate Sub-Queries**: Query OpenAI’s API to expand the original query with multiple related sub-queries.
4. **Retrieve Context**: Retrieve relevant text chunks for the joint query from the document embeddings.
5. **Expand and Refine**: Feed the augmented query back into the LLM for a **possibly** more accurate, context-rich response.

This project explores how query expansion can change the contextual understanding of the LLM, enhancing both retrieval accuracy and response relevance across different domains.

The **downside** of query expansion is that the model-generated queries might not always be relevant or useful and can sometimes introduce unnecessary noise in the final results. That is why it is important to carefully inspect the initial generated queries that will be used for query expansion. 

## Getting Started
Follow these steps to set up the project and install the necessary dependencies:

1. **Clone the repository**:
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. Install Dependencies
    Install the required dependencies using the requirements.txt file:
    ```bash
    pip install -r requirements.txt

3. Set up OpenAI API access:
    This project requires access to OpenAI’s API. Follow these steps:
	* Create an OpenAI account: Visit OpenAI’s website to sign up if you don’t already have an account.
	* Generate an API key: 
         - Go to the API Keys page in your OpenAI dashboard.
	     - Click “Create new secret key” and copy the key.
	* Purchase API usage credits: Ensure that your OpenAI account has sufficient credits or a billing plan set up for API usage.

4. Add you OpenAI API Key to the Project
    * Create a .env file in the root of the repository:
     ```bash
        touch .env
     ```
    * Add the following line to the .env file, replacing your-api-key-here with your actual OpenAI API key:
     - OPENAI_API_KEY=place-your-api-key-here

## Conclusions
The results demonstrate the effectiveness of the **Query Expansion** technique. Using UMAP visualizations:

- **Grey Dots**: Represent the embedded text of the entire document.
- **Red X**: Denotes the original query embedding.
- **Orange X**: Denotes the joint query embedding (a combination of the original query and augmented queries).
- **Green Circles**: Represent the outputs from the joint query.

The visualization clearly visualizes that the **green circles**, representing the outputs of the joint query, are significantly closer to the **orange X** (joint query embedding) compared to the **red X** (original query embedding). This indicates a significant improvement in the relevance of retrieved information after applying the joint query, thereby enhancing the contextual understanding and performance of the LLM.

The results reveal the potential of this technique to refine the retrieval process and improve response accuracy in Retrieval-Augmented Generation workflows.

