import gradio as gr
from time import sleep
from typing import List
import logging

from config import PDF_PATH, MODEL_NAME
from utils.pdf_parser import extract_text_from_pdf, split_text_into_chunks
from utils.vectorstore import build_vectorstore, load_vectorstore, retrieve_relevant_chunks
from utils.rag_pipeline import generate_answer

from sentence_transformers import SentenceTransformer

import os
os.makedirs("./data/embeddings", exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONTEXT_CHUNKS_NUM = 5

class RAGBot:
    def __init__(self):
        self.model = None
        self.chunks = []
        self.index = None
        self.is_initialized = False

    def setup(self):
        logger.info("Инициализация")
        self.model = SentenceTransformer(MODEL_NAME)
        raw_text = extract_text_from_pdf(PDF_PATH)
        self.chunks = split_text_into_chunks(raw_text)
    
        index, _, _ = build_vectorstore(self.model, self.chunks)
        
        self.index = index
        self.is_initialized = True

    def answer_question(self, question: str) -> str:
        logger.info(f"Получен вопрос: {question}")
        try:
            if self.model is None:
                logger.info("Модель не загружена, начинаем загрузку...")
                self.load_resources()
            
            logger.info("Получение релевантных чанков")
            relevant_chunks = retrieve_relevant_chunks(question, self.index, self.chunks, self.model)
            
            if not relevant_chunks:
                logger.warning("Релевантные чанки не найдены")
                return "Контекст не найден"
    
            context = "\n\n".join(relevant_chunks[:CONTEXT_CHUNKS_NUM])
            logger.info(f"Генерация ответа на основе контекста:\n{context[:200]}...")
    
            answer = generate_answer(context, question)
            return answer
    
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            return f"Произошла ошибка: {str(e)}"


bot = RAGBot()

def respond(message, chat_history):
    yield "Ищу ответ"
    sleep(1)
    result = bot.answer_question(message)
    yield f"{result}"

with gr.Blocks() as demo:
    gr.ChatInterface(
        respond,
        additional_inputs=[],
        chatbot=gr.Chatbot(label="RAG-бот по IBM SPSS", bubble_full_width=False),
        textbox=gr.Textbox(placeholder="Введите вопрос по IBM SPSS...", label="Ваш вопрос"),
        title="RAG-бот по IBM SPSS",
        description="Задайте вопрос — я найду ответ в официальной документации.",
        examples=[
            "Как запустить SPSS?",
            "Как создать таблицу в SPSS?"
        ]
    )

if __name__ == "__main__":
    logger.info("Gradio готов к запуску")
    bot.setup()
    demo.launch(ssr_mode=False, share=True)