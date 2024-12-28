Created an AI chatbot for a mock coffee shop. 
  Used data from open source kaggle to feed into my machine learning model. 
  The specific ML model I used was apriori - a recommendation based model.

Used pinecone vector database to group similar words together.
Used firebase database to host images for display in my application.

Have different agents to answer various questions:
  Guard Agent: Determines if the question is related to the coffee shop, otherwise it will dismiss it
  Classification Agent: Determines which agent the user input should pass through to
  Recommendation Agent: Determines what products to recommend based on the user's input and our apriori algorithm
  Details Agent: Gives a description about the menu, the coffee shop, and the location
  Order Agent: Keeps track of your order and price.
  
