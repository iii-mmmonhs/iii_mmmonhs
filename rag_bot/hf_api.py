import os
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("hf_api")

def call_api(prompt):
    """
    Отправляет промпт в HuggingFace Inference API и возвращает сгенерированный текст.
    
    :param prompt: входной текст для модели
    :return: сгенерированный ответ
    """
    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2-1.5B-Instruct" 
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
    
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": prompt}
    )
    
    try:
        return response.json()[0]['generated_text']
    except Exception as e:
        logger.error(f"Ошибка при получении ответа: {e}", exc_info=True)
        return "Не удалось получить ответ от модели."