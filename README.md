# Text_Summarization_tool
Create Text Summarization Tool that condenses long articles into concise summaries with a User Interface.
Python, Machine Learning, Deep Learning, HTML. 
Developed a Text Summarization Tool that condenses lengthy articles into concise summaries, featuring 
a user-friendly interface for seamless interaction.

The process of summarizing a text involves cutting it down to a shorter version while maintaining the essential factual components and the content's significance. Academic research is highly motivated to automate text summarizing since it is a time-consuming and arduous activity that is often done by hand. This trend is growing in popularity.
Text summarization has significant uses in a variety of NLP-related activities, including text classification, question answering, summarizing legal documents, news summary, and headline creation.Additionally, these systems can incorporate the creation of summaries as a transitional step that aids in document reduction.  
**# Backgorund** 
**Significance of Automatic Text Summarization (ATS):** The papers underline the growing importance of ATS in managing the vast amount of textual data available. They highlight its role in condensing lengthy texts while preserving essential information, which is crucial given the increasing volume of textual data in digital form. 
**Techniques in ATS:** The papers discuss two main categories of text summarization techniques: extractive and abstractive. Extractive methods involve selecting important sentences or phrases from the original text, while abstractive methods generate summaries by rephrasing and synthesizing key information. Recent advancements in abstractive techniques, driven by neural networks like BERT, GPT, and BART, are noted. 
**Challenges in ATS:** Challenges inherent in ATS are acknowledged, such as accurately identifying informative segments, summarizing long documents effectively, and evaluating summary quality. These challenges underscore the complexity of developing effective text summarization systems. 
**Research Frameworks and Surveys:** **Various research frameworks and surveys referenced in the papers provide comprehensive insights into ATS classifications, approaches, methods, evaluation criteria, and future research directions. These resources serve as valuable references for researchers and practitioners in the field. 
**Methodologies for Text Summarization Projects:** The papers outline methodologies for text summarization projects, including preprocessing, text representation, summarization algorithms, and user interface considerations. These methodologies provide a structured approach to developing ATS systems. 
**Evaluation Metrics:** Various evaluation metrics used in ATS, such as BLEU, ROUGE, METEOR, and G-EVAL, are discussed. These metrics play a crucial role in assessing summary quality and guiding system development. 
Overall, the papers highlight the importance of ATS in managing textual data, discuss various techniques and methodologies employed in ATS, address challenges in the field, and provide insights into future research directions. The papers discuss the importance and challenges associated with Automatic Text Summarization (ATS) systems, which are designed to condense large volumes of textual data into concise summaries while retaining essential information. These systems come in various forms, including employing extractive, abstractive, or hybrid approaches. 
Extractive methods select important sentences directly from the input paper, while abstractive methods generate summaries by rephrasing and paraphrasing content. Hybrid approaches combine both extractive and abstractive techniques for optimal results. The ATS process involves stages like pre-processing, processing, and post-processing, where linguistic techniques are applied to structure the paper, summarization techniques are used to generate the summary, and issues like sentence reordering are addressed. 
The papers also highlight the evolution of research in ATS since the 1950s, aiming to create summaries that cover main topics, avoid redundancy, and maintain cohesion. Recent surveys focus on extractive or abstractive methods, domainspecific summarization, and evaluation techniques. There is ongoing research interest in combining outputs from multiple ATS algorithms and integrating extractive and abstractive approaches into hybrid systems. 
The papers aim to aid researchers by providing a comprehensive overview of ATS classifications, approaches, methods, techniques, standard datasets, evaluation criteria, and future research directions. They address challenges such as multi-document summarization and user-specific applications, aiming to advance the field of ATS and make the technology more accessible and efficient. 
**# METHODOLOGY** 
The codes we are generating performs extractive summarization on a given text document using the cosine similarity between sentences. Here is a descriptive methodology for the code: 
**File Reading:** The code starts by defining a function read_article that takes a file path as input and reads the content of the file. The content is then returned as a string. 

**Sentence Tokenization:** The NLTK library is used to tokenize the article into sentences using nltk.sent_tokenize. Each sentence is treated as a unit for further analysis. 

**Stopwords and Lowercasing: **Stopwords are common words that do not contribute much to the meaning of the text. The code uses the NLTK library to download a list of English stopwords. Each word in the sentences is converted to lowercase to ensure case-insensitive comparison. 

**Sentence Similarity Calculation:** The function sentence_similarity is defined to calculate the similarity between two sentences using cosine distance. For each sentence pair, a vector representation is created for both sentences, and the cosine distance is calculated. The similarity is then computed as 1 - cosine_distance. 

**Build Similarity Matrix:** The function build_similarity_matrix constructs a square matrix where each cell represents the similarity score between two sentences. It uses the sentence_similarity function for pairwise comparison of sentences. 

**Score Calculation:** The code calculates a score for each sentence based on its similarity with other sentences. The scores are stored in the scores array. 
 
**Top-N Sentence Selection: **The top-n sentences are selected based on their scores. The code uses np.argsort(scores) to get the indices of sentences sorted in descending order of their scores. The top-n sentences are then selected and stored in the ranked_sentences list. 
 
**Summary Generation: **The selected top-n sentences are joined together to form a summary using ' '.join (ranked_sentences). The generated summary is then printed. 
 
**Further Analysis or Visualization: **The code mentions that the sentence_similarity_matrix can beused for further analysis or visualization. This matrix contains the pairwise similarity scores between sentences. 
 
**Adjustable Parameters: **The top_n variable allows users to adjust the number of sentences in the final summary based on their preference. 

**# ALGORITHM **
Extractive Text Summarization using Cosine Similarity 
**1. Preprocessing: **
Tokenization: Break the input text into individual words and sentences using NLTK's sent_tokenize function. 
Stopword Removal: Remove common stopwords (e.g., 'is', 'the', 'and') using NLTK's English stopwords list. 
**2.  Sentence Similarity: **
Define a function sentence_similarity to calculate the similarity between two sentences. For each sentence pair, create vectors representing the frequency of each word in the sentences. Compute the cosine similarity between the vectors to determine the similarity score. 
**3. Similarity Matrix: **
Build a similarity matrix where each cell represents the similarity score between two sentences. Iterate through all pairs of sentences and calculate their similarity scores using the sentence_similarity function. 
**4. Sentence Ranking: **
Calculate the total similarity score for each sentence by summing its similarity scores with all other sentences.Rank the sentences based on their total similarity scores. 
**5. Summary Generation: **
Select the top-N ranked sentences to form the summary. Combine the selected sentences to generate the final summary. 
**# Theoretical Explanation **
Cosine Similarity: Cosine similarity measures the cosine of the angle between two vectors in a multidimensional space. In this context, the vectors represent the frequency of words in sentences. Higher cosine similarity indicates greater similarity between sentences. 
Extractive Summarization: Extractive summarization involves selecting a subset of sentences from the original text to construct the summary. The algorithm selects the most important sentences based on their similarity to other sentences in the text. 
Evaluation Metrics: To evaluate the quality of the summary, metrics such as ROUGE scores can be used. ROUGE measures the overlap between the generated summary and reference summaries provided by human annotators. 
