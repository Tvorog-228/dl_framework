import os
import cv2
import numpy as np
from Tensor import Tensor
import Layer as L
from SGD import SGD

def load_data(dir,img_size=64, limit=1000):
    data=[]
    labels=[]
    categorias=['NORMAL', 'PNEUMONIA']

    for categoria in categorias:
        path=os.path.join(dir, categoria)
        label=categorias.index(categoria)
        i=0

        for img_name in os.listdir(path):
            if i==limit:
                break
            i+=1

            try:
                img_path=os.path.join(path, img_name)
                img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                else:
                    print(f"Error: No se pudo cargar la imagen en {img_path}")
                    continue

                #(píxeles de 0-255 a 0-1)
                data.append(img.flatten() / 255.0)
                labels.append(label)
            except Exception as e:
                pass
    return np.array(data), np.array(labels)


def evaluate(model, x_test, y_test):
    test_x = Tensor(x_test, autograd=False)

    predictions_out = model.forward(test_x)

    preds = np.argmax(predictions_out.data, axis=1)

    accuracy = (preds == y_test).mean() * 100
    return accuracy, preds

def train():
    x_train, y_train = load_data('archive/chest_xray/train')
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
    print(x_train.shape)
    print(y_train.shape)


    input_size=64*64
    hidden_size=256
    output_size=2

    model = L.Sequential()
    model.add(L.Linear(input_size, hidden_size))
    model.add(L.Tanh())
    model.add(L.Linear(hidden_size, output_size))

    criterion = L.CrossEntropyLoss()

    optim = SGD(parameters=model.get_parameters(), alpha=0.01)

    batch_size = 16
    epochs = 100


    for epoch in range(epochs):
        total_loss = 0

        for i in range(0, len(x_train), batch_size):
            batch_x = Tensor(x_train[i : i + batch_size], autograd=False)
            batch_y = Tensor(y_train[i : i + batch_size], autograd=False)

            optim.zero()

            prediction = model.forward(batch_x)

            loss = criterion.forward(prediction, batch_y)

            loss.backward()

            optim.step()

            total_loss += loss.data

        print(f"Epoch {epoch} - Loss: {total_loss / (len(x_train)/batch_size):.4f}")

    #test
    x_test, y_test = load_data('archive/chest_xray/test', limit=500)

    print("Evaluando modelo...")
    acc, predictions = evaluate(model, x_test, y_test)

    print(f"\n" + "="*30)
    print(f"RESULTADO FINAL")
    print(f"Accuracy en Test: {acc:.2f}%")
    print("="*30)

    print("\nMuestra de predicciones (Primeras 10):")
    nombres = ['NORMAL', 'PNEUMONIA']
    for i in range(10):
        real = nombres[y_test[i]]
        pred = nombres[predictions[i]]
        print(f"Real: {real} | Predicción: {pred}")

if __name__ == '__main__':
    train()



