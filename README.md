# Mistral-AI
Many popular and very effective LLMs are built on the standard transformer architecture, but one of the open source models released by **Mistral (Mistral 8x7B)** modifies the standard transformer architecture using a **mixture of experts**. There are **8 distinct Feed Forward Neural Networks** called experts, and at inference time a different **Gating Neural Network** first chooses to activate two of these eight experts to run to predict the next token. It then takes the **weighted average** of these to expert outputs in order to actually generate that next token. This mixture of expert design allows the Mistral model to have both the **performance improvements** of larger models while having **inference costs** comparable to a smaller model. Even though the Mixtral model has **46.7B parameters** at inference time, it only uses **12.9B** of those parameters to predict each token.<br>
When we integrate an LLM into a larger software application, it is often very helpful for the LLM's output to be easily fed into downstream software systems by having it open as response in a structured JSON format. For some LLMs users may rely on clever prompting or using a framework like **LangChain-a-LlamaIndex** to guarantee a reliable JSON format in the response. Mixtral has a reliable feature to generate responses in the JSON format that we request. 

## Overview
Mistral offers **6 models** for all use cases and business needs that we can download their weights and use it anywhere without any restrictions. **Mistral 7B** which fits on one GPU outperforms **LLaMa model** with similar and even greater sizes. **Mistral 8x7B** is a sparse mixture of expert models. The foundation of this model is a transformer block consisting of 2 layers, feed forward layer and multi head attention layer. Each input token goes through the same layers. We duplicate the feed forward layer N times. To decide which input token goes to which layer we use a router to map each token to the top K feed forward layers and ignore the rest. As a result, even though Mistral has **46.7B parameters**, it only uses **12.9B parameters** per token providing great performance with fast inference. It outperforms **LLaMa 2.70B** and most benchmarks with **8x** faster inference and it matches or outperforms **GPT 3.5** at most standard benchmarks. These models are under the open source **Apache 2.0 License** means we can download the model weights of both models, fine tune and customize them for our own use cases and use them without any restrictions.<br>
Mistral also offers 4 optimized enterprise-grade models. **Mistral Small** is best for lower latency use cases, **Mistral Medium** is suitable for our language based tasks, **Mistral Large** is the flagship model for our most sophisticated needs with advanced reasoning capabilities, it approaches the performance of **GPT-4** and outperforms also has native multilingual capabilities, it offers a **32K tokens** context window. Finally the Embedding Model which offers the SOTA embeddings for text and can be used for many use cases like clustering and classification. We can chat with the Mistral models in **chat.mistral.ai**. To use these models in code we can use **transformers, llama.cpp** or **ollama**. To setup the API key goto **console.mistral.ai**.

## 1. Prompting Capabilities
Prompt the mistral models via API calls and perform various tasks like classification information extraction, personalization and summarization.<br>
`!pip install mistralai<br>
from helper import load_mistral_api_key<br>
load_mistral_api_key()<br>`

`from helper import mistral`<br>
`mistral("hello, what can you do?")`<br>
`"Hello! I'm here to help answer your questions, provide information, offer explanations, and even share a joke or two. I can assist with a wide range of topics, including but not limited to, general knowledge, science, history, literature, math, and more. I can also help explain concepts, solve problems, and offer guidance on various subjects. What can I assist you with today?"`

### Classification
`prompt = """
    You are a bank customer service bot. 
    Your task is to assess customer intent and categorize customer 
    inquiry after <<<>>> into one of the following predefined categories:
    
    card arrival
    change pin
    exchange rate
    country support 
    cancel transfer
    charge dispute
    
    If the text doesn't fit into any of the above categories, 
    classify it as:
    customer service
    
    You will only respond with the predefined category. 
    Do not provide explanations or notes. 
    
    ###
    Here are some examples:
    
    Inquiry: How do I know if I will get my card, or if it is lost? I am concerned about the delivery process and would like to ensure that I will receive my card as expected. Could you please provide information about the tracking process for my card, or confirm if there are any indicators to identify if the card has been lost during delivery?
    Category: card arrival
    Inquiry: I am planning an international trip to Paris and would like to inquire about the current exchange rates for Euros as well as any associated fees for foreign transactions.
    Category: exchange rate 
    Inquiry: What countries are getting support? I will be traveling and living abroad for an extended period of time, specifically in France and Germany, and would appreciate any information regarding compatibility and functionality in these regions.
    Category: country support
    Inquiry: Can I get help starting my computer? I am having difficulty starting my computer, and would appreciate your expertise in helping me troubleshoot the issue. 
    Category: customer service
    ###
    
    <<<
    Inquiry: {inquiry}
    >>>
    Category:
"""`<br>
`In the above prompt our task is to assess customer intent and categorize customer inquiry. We have a list of predefined categories. If the text doesn't fit in any of the categories, classify it as customer service.
In the above prompt we first assign a **role play** to the model as a `bank customer service bot` this adds personal context to the model. Next we used **few shot learning** where we give some exapmples in the prompts. This can improve model's performance, especially when the task is difficult or when we want the model to respond in a specific manner.
We use demminetors like **'###'** and **'<<<'** to specify the boundary between diferent sections of the text. In our example, we use the '###' to indicate examples and '<<<' to indicate customer inquiry.
Finally in a case when the model is **verbose**, we can add: "do not provide explanations or notes", to make sure the output is concise.`

