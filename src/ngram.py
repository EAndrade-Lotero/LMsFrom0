import torch
import torch.nn as nn
import torch.nn.functional as F




CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Star Wars from wikipedia 
test_sentence = """Star Wars, conocida también en español como La guerra de las galaxias,
[1]​[2]​[3]​ es una franquicia y universo compartido de fantasía compuesta primordialmente de una serie de 
películas concebidas por el cineasta estadounidense George Lucas en la década de 1970, y producidas y 
distribuidas inicialmente por 20th Century Fox y posteriormente por The Walt Disney Company a partir de 2012. 
Su trama describe las vivencias de un grupo de personajes que habitan en una galaxia ficticia e interactúan 
con elementos como «la Fuerza», un campo de energía metafísico y omnipresente[4]​ que posee un «lado luminoso» 
impulsado por la sabiduría, la nobleza y la justicia y utilizado por los Jedi, y un «lado oscuro» usado por 
los Sith y provocado por la ira, el miedo, el odio y la desesperación.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# Print the first 3, just so you can see what they look like.
print('First ngrams',ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in ngrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!

# To get the embedding of a particular word, e.g. 
print(model.embeddings.weight[word_to_ix["galaxia"]])