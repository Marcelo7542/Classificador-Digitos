# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:47:56 2024

@author: marce
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import matplotlib.image as matim
import Module as md

Entrada, Saida = load_digits(return_X_y=True)
print(load_digits().images[0])

plt.figure(figsize=(2,2))
plt.imshow(load_digits().images[0], cmap= plt.cm.gray_r)

entradatreino, entradateste, saidatreino, saidateste = train_test_split(Entrada, Saida, test_size=0.1, random_state=2)

imagem = matim.imread(r'C:\Users\marce\OneDrive\Documentos\AI\Digitos\Classificador-Digitos\5 (1).png')

rgbimagem = md.rgb_gray(imagem)

modelo = SVC()
modelo.fit(entradatreino, saidatreino)
predicao = modelo.predict(entradateste)
acerto = metrics.accuracy_score(saidateste, predicao)
previsor = modelo.predict([rgbimagem])
print(f"A margem de acertos usando Suport vector machine: {acerto}%")
print(f"{previsor[0]} é o número na imagem")

modelo2 = LogisticRegression(max_iter=1000)
modelo2.fit(entradatreino, saidatreino)
previsao_2 = modelo2.predict(entradateste)
acerto_2 = metrics.accuracy_score(saidateste, previsao_2)
previsao_2rgb = modelo2.predict([rgbimagem])
print(f"A margem de acertos usando LogisticRegression: {acerto_2}%")
print(f"{previsao_2rgb[0]} é o número na imagem")


modelo3 = SGDClassifier(max_iter=1200)
modelo3.fit(entradatreino, saidatreino)
previsao_3 = modelo3.predict(entradateste)
acerto_3 = metrics.accuracy_score(saidateste, previsao_3)
previsao_3rgb = modelo3.predict([rgbimagem])
print(f"A margem de acertos usando SGDClassifier: {acerto_3}%")
print(f"{previsao_3rgb[0]} é o número na imagem")
