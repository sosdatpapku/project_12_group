import io
import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

# Токенизатор и модель взята с hugginface "https://huggingface.co/blanchefort/rubert-base-cased-sentiment"
tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment', return_dict=True)

# Ниже описана функция метода predict 
#@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted

def tokens(predicted): # фунция-интерпретаотр получаемых значений на выходе модели 
    if predicted == 0:
        return st.write("Результат обработки - 0: Нейтральный текст")
    elif predicted == 1:
        return st.write("Результат обработки - 1: Позитивный текст")
    else:
        return st.write("Результат обработки - 2: Негативный текст")

def main():
	st.title("Sentiment Analysis NLP App")
	st.subheader("Streamlit Projects")

	menu = ["Home","About"] # Демо реализация двустраничной версии приложения (с моделью и описанием)
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home": # выбор базовой страницы приложения с самой моделью
		st.subheader("Home")
		with st.form(key='nlpForm'): # cоздаем поле для ввода текста
			raw_text = st.text_area("Enter Text Here")
			submit_button = st.form_submit_button(label='Analyze') # кнопка подтверждения

		col1,col2 = st.columns(2)
		if submit_button:

			with col1: 
				st.info("Results")
				sentiment = predict(raw_text) # запуск модели
				#st.write(f"Результат классфикации текста {sentiment}")
				tokens(sentiment)
                

	else:
		st.subheader("About") # переход на другую страницу приложения


if __name__ == '__main__': # тело программы
	main()