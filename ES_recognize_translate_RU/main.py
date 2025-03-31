import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, MarianMTModel, MarianTokenizer

def load_models():
    processor = TrOCRProcessor.from_pretrained('qantev/trocr-large-spanish')
    model = VisionEncoderDecoderModel.from_pretrained('qantev/trocr-large-spanish')
    translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-ru")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-ru")
    return processor, model, translator_model, tokenizer

def recognize_text(image, processor, model):
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def translate_text(text, translator_model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_ids = translator_model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
    return translated_text

def main():
    st.title("Распознавание и перевод испанского текста")
    st.write("Загрузите изображение с испанским текстом, получите распознанный текст и его перевод на русский.")

    processor, model, translator_model, tokenizer = load_models()
    
    uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
        
        with st.spinner("Распознаем текст..."):
            recognized_text = recognize_text(image, processor, model)
            st.success("Текст распознан:")
            st.write(recognized_text)
        
        with st.spinner("Переводим текст..."):
            translated_text = translate_text(recognized_text, translator_model, tokenizer)
            st.success("Перевод на русский:")
            st.write(translated_text)

if __name__ == "__main__":
    main()
