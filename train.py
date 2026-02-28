import numpy as np
import sys
from Tensor import Tensor
import Layer as L
from SGD import SGD

#ajustar el sistema
sys.setrecursionlimit(10000)
np.random.seed(0)

#cargar datos
try:
    with open('Shakespear.txt', 'r') as f:
        raw = f.read()
except:
    raw = "shakespeare sample text for the recurrent neural network training " * 500

vocab = list(set(raw))
word2index = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)
indices = np.array([word2index[x] for x in raw])

#configurar el modelo
batch_size = 32
bptt = 16
n_factors = 256

embed = L.Embedding(vocab_size=vocab_size, dim=n_factors)
model = L.LSTMCell(n_inputs=n_factors, n_hidden=n_factors, n_output=vocab_size)
model.w_ho.weight.data *= 0

criterion = L.CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)

#prepara batches
n_batches = int(indices.shape[0] / batch_size)
trimmed_indices = indices[:n_batches * batch_size]
batched_indices = trimmed_indices.reshape(batch_size, n_batches).transpose()

input_batched_indices = batched_indices[0:-1]
target_batched_indices = batched_indices[1:]

n_bptt = int(((n_batches - 1) / bptt))
input_batches = input_batched_indices[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)
target_batches = target_batched_indices[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)

def generate_sample(n=30, init_char='\n'):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    if init_char not in word2index: init_char = vocab[0]
    input_idx = Tensor(np.array([word2index[init_char]]), autograd=False)

    for i in range(n):
        rnn_input = embed.forward(input_idx)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
        m = output.data.argmax()

        input_idx = Tensor(np.array([m]), autograd=False)
        s += vocab[m]

        hidden = (Tensor(hidden[0].data, autograd=True),Tensor(hidden[1].data, autograd=True))
    return s

#entrenar
def train(iterations=100):
    min_loss = 1000
    print(f"Iniciando entrenamiento por {iterations} iteraciones...\n")

    for iter in range(iterations):
        total_loss = 0
        hidden = model.init_hidden(batch_size=batch_size)

        for batch_i in range(len(input_batches)):
            hidden = (Tensor(hidden[0].data, autograd=True), Tensor(hidden[1].data, autograd=True))

            optim.zero()
            losses = list()

            for t in range(bptt):
                input_t = Tensor(input_batches[batch_i][t], autograd=False)
                rnn_input = embed.forward(input=input_t)
                output, hidden = model.forward(input=rnn_input, hidden=hidden)

                target_t = Tensor(target_batches[batch_i][t], autograd=False)
                batch_loss = criterion.forward(output, target_t)

                if t == 0: losses.append(batch_loss)
                else: losses.append(batch_loss + losses[-1])

            loss = losses[-1]
            loss.backward()

            for p in (model.get_parameters() + embed.get_parameters()):
                if p.grad is not None:
                    p.grad = np.clip(p.grad, -1, 1)

            optim.step()

            total_loss += loss.data / bptt
            epoch_loss = np.exp(total_loss / (batch_i + 1))

            if epoch_loss < min_loss:
                min_loss = epoch_loss

            log = f"\r Iter:{iter} | Batch {batch_i+1}/{len(input_batches)} | Loss:{epoch_loss:.4f}"
            sys.stdout.write(log)

        if iter % 10 == 0:
            print(f"\n\n--- MUESTRA ITERACIÓN {iter} (n=250) ---")
            print(generate_sample(n=250, init_char='\n'))
            print("-" * 40 + "\n")

        optim.alpha *= 0.99

    print("\n\n" + "="*50)
    print("ENTRENAMIENTO FINALIZADO")
    print("--- MUESTRA FINAL DE 1000 CARACTERES ---")
    print(generate_sample(n=1000, init_char='\n'))
    print("="*50)

if __name__ == "__main__":
    train(iterations=100)
