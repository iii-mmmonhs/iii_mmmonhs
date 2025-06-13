import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VectorStore")

CHUNKS_EXT = ".pkl"


def build_vectorstore(model: SentenceTransformer, chunks: list, save_path: str = "embeddings/index.faiss") -> None:
    """
    Строит FAISS индекс на основе эмбеддингов чанков и сохраняет его

    :param model: модель для энкодинга текста
    :param chunks: список текстовых чанков
    :param save_path: путь для сохранения индекса
    """

    logger.info("Начинается построение vectorstore")

    try:
        embeddings = model.encode(chunks, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"Ошибка при кодировании чанков: {e}", exc_info=True)
        raise

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    try:
        index.add(embeddings)
        faiss.write_index(index, save_path)
    except Exception as e:
        logger.error(f"Ошибка при сохранении индекса: {e}", exc_info=True)
        raise

    chunks_path = save_path + CHUNKS_EXT
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    logger.info(f"Сохранено {len(chunks)} чанков")


def load_vectorstore(model: SentenceTransformer, load_path: str = "embeddings/index.faiss") -> tuple:
    """
    Загружает FAISS индекс и соответствующие ему чанки

    :param model: модель, которую нужно использовать
    :param load_path: путь к файлу индекса

    Возвращает (index, chunks, model)
    """

    logger.info("Загружаем vectorstore...")

    try:
        index = faiss.read_index(load_path)
    except Exception as e:
        logger.error(f"Не удалось прочитать индекс: {e}", exc_info=True)
        raise

    chunks_path = load_path + CHUNKS_EXT
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Файл с чанками не найден: {chunks_path}")

    try:
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
    except Exception as e:
        logger.error(f"Не удалось загрузить чанки: {e}", exc_info=True)
        raise

    logger.info(f"Восстановлено {len(chunks)} чанков")
    return index, chunks, model


def retrieve_relevant_chunks(query: str, index: faiss.Index, chunks: list, model: SentenceTransformer, top_k: int = 3) -> list:
    """
    Возвращает наиболее релевантные чанки для заданного запроса

    :param query: запрос пользователя
    :param index: FAISS индекс
    :param chunks: список чанков
    :param model: модель для энкодинга
    :param top_k: количество возвращаемых чанков (топ-k релевантных)

    Возвращает список релевантных чанков
    """

    if not query or not isinstance(query, str):
        raise ValueError("Query пуст")

    try:
        q_emb = model.encode([query])
        distances, indices = index.search(np.array(q_emb), top_k)
    except Exception as e:
        logger.error(f"Ошибка при поиске релевантных чанков: {e}", exc_info=True)
        raise

    results = [chunks[i] for i in indices[0]]
    filtered = [r for r in results if len(r.split()) > 50]
    
    return filtered or results[:1]