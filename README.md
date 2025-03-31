# Распознавание и перевод рукописного испанского текста

Этот проект использует модель TrOCR для распознавания текста на испанском языке с изображений и переводит его на русский язык с помощью модели MarianMT. Все операции происходят в веб-приложении, созданном с использованием Streamlit.

## Описание

Проект реализует следующее:

1. **Распознавание текста**: Используется модель `qantev/trocr-large-spanish` для извлечения текста с изображений, содержащих рукописный испанский текст.
2. **Перевод текста**: После распознавания текста, он переводится на русский язык с использованием модели `Helsinki-NLP/opus-mt-es-ru`.

Процесс работы:
1. Загрузите изображение с испанским текстом.
2. Приложение распознает текст с изображения.
3. После этого текст переводится на русский язык.

