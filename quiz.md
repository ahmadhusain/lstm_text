1.  When building an architecture neural network with a large number of hidden layer. We may face a problem, the update value from backpropogation steps will exponentially decrease when heading to the input layer. So the model fails to learn when doing a pass forward to make a predictions. Which of the following statement is appropriate to that problem?
    - [ ] Exploding Gradient Descent
    - [ ] Vanishing Gradient Descent
    - [ ] Backpropogarion Through Time
2.  *Long Short-Term memory* (LSTM) architecture using **gate mechanism** to overcome a gradient descent's problem. Which gate that aims to controls how many internal states (information) want to exposes to the next time step?
    - [ ] Output gate
    - [ ] Hidden gate
    - [ ] Input gate
3.  When working with text dataset on classification task, there is *tokenization* step that aims to separate each word in the entire document into a token form. We need to do *feature selection* before modeling. Which of the argument on `text_tokenizer` below is appropriate to setting the maximum number of words to be used?
    - [ ] `lower`
    - [ ] `num_words`
    - [ ] `fit_text_tokenizer`
4.  If we want to create matrix result of the transformation each text in a sequence of integer, which of the function below is appropriate?
    - [ ] `text_tokenizer()`
    - [ ] `pad_sequences()`
    - [ ] `text_to_sequences()`
5.  When building an architecture model, we want a model does not "*memorize*" the training data, but learns the actual relationship on data! Which of the following methods can we use to avoid underfitting and overfitting?
    - [ ] added `layer_dropout` with dropout rate closer to 1.
    - [ ] added `layer_dropout` with dropout rate between 0.2 to 0.5
    - [ ] increase number of epoch when train the model.
6.  Suppose we want to predict unseen text data, of course we need to do pre-processing step first. Which of information from model that we need to keep when doing text-processing? 
    - [ ] tokenizer
    - [ ] epoch
    - [ ] batch size
  