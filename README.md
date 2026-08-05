# Customer support chatbot

Components of the system:

- A document ingestion pipeline, that runs on demand. It pushes new document pdfs into the knowledge database.
- A real-time chatbot that receives incoming messages from external end-users and tries to answer them faithfully based on the data in the knowledge database.


## Document ingestion pipeline

A Python service under src/customer_support_chatbot/ingestion that can embed input pdfs into a knowledge database, that can be later queried