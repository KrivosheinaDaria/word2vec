import numpy as np
import string
import re
import spacy
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from models.Net import Net
from models.Word2Vec import Word2Vec

translator = str.maketrans('', '', string.punctuation + '«»“”‘’')
nlp = spacy.load("ru_core_news_sm")

# обучение модели
def train(model, loader, criterion, optimizer, num_epoch):
    all_loss = []
    all_perplexity = []
    for t in tqdm(range(num_epoch)):
        total_loss = 0.0
        total_samples = 0

        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)

        epoch_loss = total_loss / total_samples
        epoch_perplexity = np.exp(epoch_loss)
        all_perplexity.append(epoch_perplexity)
        all_loss.append(epoch_loss)
        print('Epoch {} \t'.format(t), 'Loss: {}'.format(np.round(epoch_loss, 6)))
        print('Epoch {} \t'.format(t), 'Perplexity: {}'.format(np.round(epoch_perplexity, 6)))
    return model

def string_processing(input_string):
    input_string = re.sub(r'\d+', '', input_string)
    input_string = input_string.translate(translator).split()
    input_string = [word for word in input_string if len(word) >= 3]
    input_string = [nlp(word)[0].lemma_ for word in input_string]
    input_string = input_string[-obj.L_size:]
    vec = np.zeros(obj.dictionary_size)
    for i in input_string:
        vec[obj.dictionary_words[i]] = 1
    vec = np.matmul(vec, embedding)
    return vec

if __name__ == '__main__':
    file = open('sholohov-don.txt', 'r', encoding='utf-8')
    text = file.read()

    # удаление чисел
    text = re.sub(r'\d+', '', text)

    # удаление знаков препинания
    text_changed = text.translate(translator).lower().split()[1000:4900]

    # лемматизация
    text_changed = [nlp(word)[0].lemma_ for word in text_changed if len(word) >= 3]

    # обучение эмбеддинга на тексте
    obj = Word2Vec(text_changed, 4, 500)
    embedding = obj.train(10, 0.1)

    # подготовка данных (применение эмбеддинга к каждому контексту)
    x_train = []
    y_train = []
    for x, y in obj.dictionary_context.items():
        x_vector = np.zeros(obj.dictionary_size)
        for k in range(len(x)):
            x_vector[x[k]] = 1
        x_vector = np.matmul(x_vector, embedding)
        x_train.append(x_vector)
        y_train.append(y)

    # создание загрузчиков
    x_train = torch.FloatTensor(np.array(x_train))
    y_train = torch.LongTensor(np.array(y_train))
    train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=64, shuffle=True)

    # создание модели
    model = Net(obj.vector_size, obj.dictionary_size)

    # обучение модели
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(model, train_loader, criterion, optimizer, 120)

    # тестирование модели
    str1 = 'Решило исход с Колчаком и Деникиным'
    str1 = string_processing(str1)
    output = model(torch.tensor(str1))
    print(obj.dictionary_indexes[list(output).index(max(list(output)))])

    str2 = 'Семь собственников одновременно взвесили'
    str2 = string_processing(str2)
    output = model(torch.tensor(str2))
    print(obj.dictionary_indexes[list(output).index(max(list(output)))])

    str3 = 'Казак издавна трудится за семью'
    str3 = string_processing(str3)
    output = model(torch.tensor(str3))
    print(obj.dictionary_indexes[list(output).index(max(list(output)))])

    str4 = 'Долгий год дал основание'
    str4 = string_processing(str4)
    output = model(torch.tensor(str4))
    print(obj.dictionary_indexes[list(output).index(max(list(output)))])

    str5 = 'Матрос на улице упрекал злодея'
    str5 = string_processing(str5)
    output = model(torch.tensor(str5))
    print(obj.dictionary_indexes[list(output).index(max(list(output)))])


