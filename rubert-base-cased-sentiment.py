!pip install transformers sentencepiece sacremoses

from transformers import pipeline

classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")

lis = ['В начале 1960-х корпорация IBM была абсолютным лидером рынка компьютеров','Однако перспективы становились все менее радужными', 'Да и для самой IBM ситуация выглядела плохо.']
classifier(lis)
classifier('Мне очень плохо')
classifier('Мне очень хорошо')
