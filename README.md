# Pulpnet-ACA-
A Transformer-based chatbot designed to answer user queries using data scraped from the IIT Kanpur academic website. The model leverages a pre-trained RoBERTa transformer for extractive QA and offers a smooth, user-friendly interface built with Streamlit.

*Access the app at*: https://nishtha-pulpnet-project.streamlit.app/

***Workflow Overview***
1. **Web Scraping** – Extracts textual data from IIT Kanpur's official web pages      
2. **Text Cleaning & Filtering** – Removes irrelevant/duplicate content   
3. **QA Model Setup** – Loads a pretrained transformer model (FLAN-T5 base) from Hugging Face for extractive question answering.
4. **Chatbot Deployment** – A Streamlit-based UI lets users interact with the chatbot

**Data Collection**

You can either:
1. Run the Jupyter notebook:```Data_Scrapping_IITK.ipynb``` to scrape the latest data from IIT Kanpur's website.

2. Or use the pre-scraped, cleaned dataset:```iitk_cleaned_data.csv```


**Model Selection**

We use a pre trained model FLAN-T5 base from Hugging Face's Transformers library 

**Streamlit Deployment**

*To run locally*

 1. Clone the Repository
   
    ```bash
    git clone https://github.com/gnishtha05/Pulpnet-ACA-.git
    ```
 2. Import dependencies
      
    ```bash
    pip install -r requirements.txt
    ```
 3. Run the following commands to launch the chatbot
      
    ```bash
    cd Pulpnet-ACA-
    streamlit run final_project/app.py
    ```

*Hosted link*: https://nishtha-pulpnet-project.streamlit.app/






Submisson By - Nishtha Gupta

Roll No. - 240708

