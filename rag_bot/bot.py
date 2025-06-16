import os
import sys
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("Bot")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

PROJECT_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
logger.info(f"Рабочая директория: {PROJECT_DIR}")

sys.path.append(str(PROJECT_DIR))

try:
    import imghdr
except ImportError:
    import sys
    sys.modules['imghdr'] = type(sys)(name='imghdr')

import asyncio
from telegram import Update
from telegram.ext import (ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters)
from sentence_transformers import SentenceTransformer

from config import PDF_PATH, EMBEDDINGS_PATH, CHUNK_SIZE, OVERLAP
from utils.pdf_parser import extract_text_from_pdf, split_text_into_chunks
from utils.vectorstore import build_vectorstore, load_vectorstore, retrieve_relevant_chunks
from utils.rag_pipeline import generate_answer


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Не найден TELEGRAM_TOKEN. Пожалуйста, проверьте его наличие в файле .env.")

async def initialize_data():
    """
    Инициализирует данные при старте бота
    Загружает или строит FAISS индекс и чанки
    """
    global index, chunks, model
    logger.info("Подготовка данных...")

    if not os.path.exists(EMBEDDINGS_PATH):
        logger.info("Строим векторное хранилище...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        raw_text = extract_text_from_pdf(PDF_PATH)
        chunks = split_text_into_chunks(raw_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

        build_vectorstore(model, chunks, EMBEDDINGS_PATH)

    else:
        logger.info("Загружаем существующее хранилище...")
        index, chunks, model = load_vectorstore(EMBEDDINGS_PATH)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды /start
    """
    await update.message.reply_text(
        "Привет! Я Ваш помощник по IBM SPSS. Задайте вопрос, а я найду ответ в документации."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global index, chunks, model

    try:
        query = update.message.text.strip()
        if not query:
            return

        logger.info(f"Получено сообщение: {query[:100]}...")

        relevant_chunks = retrieve_relevant_chunks(query, index, chunks, model)
        if not relevant_chunks:
            await update.message.reply_text("Контекст не найден.")
            return

        context_str = "\n\n---\n\n".join(relevant_chunks[:2])
        logger.info(f"Используемый контекст:\n{context_str[:500]}...")

        answer = generate_answer(context_str, query)

        MAX_LEN = 4096
        messages = [answer[i:i + MAX_LEN] for i in range(0, len(answer), MAX_LEN)]

        for msg in messages:
            await update.message.reply_text(msg or "Извините, не могу ответить на Ваш вопрос.")

    except Exception as e:
        logger.error(f"Ошибка в handle_message: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка. Попробуйте еще раз.")

async def run_bot():
    logger.info("Бот запускается...")
    await initialize_data()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен. Ждём сообщения...")
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_bot())