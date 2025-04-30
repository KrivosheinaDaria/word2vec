import numpy as np

class Word2Vec:
    def __init__(self, words, L_size, vector_size):
        # размер контекста
        self.L_size = L_size
        # размер эмбеддинга
        self.vector_size = vector_size

        # словари слово -> индекс и индекс -> слово
        dictionary_words = {}
        dictionary_indexes = {}
        for i in range(len(list(words))):
            dictionary_words[words[i]] = i
            dictionary_indexes[i] = words[i]
        self.dictionary_words = dictionary_words
        self.dictionary_indexes = dictionary_indexes
        self.dictionary_size = len(list(words))

        # матрицы преобразования слов в векторы (= эмбеддинги)
        self.W = np.random.rand(self.dictionary_size, self.vector_size)
        self.C = np.random.rand(self.vector_size, self.dictionary_size)

        # словарь контекст -> целевое слово
        dictionary_context = {}
        for i in range(len(words)):
            context = []
            for j in range(max(0, i - int(self.L_size / 2)), min(len(words), i + int(self.L_size / 2) + 1)):
                if i != j:
                    context.append(self.dictionary_words[words[j]])
            dictionary_context[tuple(set(context))] = self.dictionary_words[words[i]]
        self.dictionary_context = dictionary_context

    def cross_entropy_loss(self, prediction, target):
        return -np.sum(target * np.log(prediction + 1e-10))

    def softmax(self, vector):
        exponent = np.exp(vector - np.max(vector))
        return exponent / np.sum(exponent)

    def train(self, epochs, learning_rate):
        for epoch in range(epochs):
            loss = 0
            for context, target in self.dictionary_context.items():
                # суммирование one-hot векторов контекстных слов
                context_vector = np.zeros((1, self.dictionary_size))
                for k in range(len(context)):
                    context_vector[0][context[k]] = 1

                # применение матриц-эмбеддингов
                step1 = np.matmul(context_vector, self.W)
                step2 = np.matmul(step1, self.C)

                softmax = self.softmax(step2)

                target_vector = np.zeros((1, self.dictionary_size))
                target_vector[0][target] = 1

                # расчет лосса
                loss += self.cross_entropy_loss(softmax, target_vector)

                # обновление весов
                gradient = softmax - target_vector
                for i in range(self.vector_size):
                    for j in range(self.dictionary_size):
                        self.C[i][j] -= learning_rate * context_vector[0][i] * gradient[0][j]
                for i in context:
                    self.W[i] -= learning_rate * np.dot(self.C, gradient[0])
            print('Epoch {} loss: {}'.format(epoch, loss))
        return self.W
