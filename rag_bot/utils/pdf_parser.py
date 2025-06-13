import fitz
import re
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PDFParser")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из PDF-файла

    :param pdf_path: путь к PDF-файлу

    Возвращает строку с текстом
    """
    logger.info(f"Извлечение текста из PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        logger.error(f"Файл не найден: {pdf_path}", exc_info=True)
        raise FileNotFoundError(f"Файл не найден в директории: {pdf_path}")

    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()

        logger.info(f"Текст успешно извлечён (длина: {len(text)} символов)")
    except Exception as e:
        logger.error(f"Ошибка при чтении PDF: {e}", exc_info=True)
        raise RuntimeError(f"Ошибка при чтении PDF: {e}")

    return clean_text(text)


def clean_text(text: str) -> str:
    """
    Очищает текст от лишних пробелов и символов

    :param text: сырой текст

    Возвращает очищенный текст
    """
    logger.debug("Начинаем очистку текста")

    cleaned_text = re.sub(r'\s+', ' ', text)

    logger.debug(f"Текст до очистки: {len(text)} символов, после: {len(cleaned_text)} символов")
    return cleaned_text.strip()


def split_text_into_chunks(text: str, chunk_size: int = 512, overlap: int = 32) -> list:
    """
    Делит текст на чанки с перекрытием

    :param text: входной текст
    :param chunk_size: размер одного чанка в словах
    :param overlap: количество слов, которые будут перекрываться между чанками

    Возвращает список чанков
    """
    logger.info(f"Начинаем разбиение текста на чанки (размер={chunk_size}, перекрытие={overlap})")

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        end = i + chunk_size + overlap
        chunk = " ".join(words[i:end])
        chunks.append(chunk)

    logger.info(f"Создано {len(chunks)} чанков")
    return chunks