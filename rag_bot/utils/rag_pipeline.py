import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RAGPipeline")


def load_generator(model_name="Qwen/Qwen2-1.5B-Instruct"):
    """
    Загружает модель и токенизатор, создаёт пайплайн генерации
    
    :param model_name: имя модели на HuggingFace

    Возвращает (generator, tokenizer)
    """
    logger.info(f"Загружаем генеративную модель: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            return_full_text=False
        )

        logger.info("Модель загружена")
        return generator, tokenizer

    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}", exc_info=True)
        raise


def truncate_context(tokenizer, context: str, max_length: int = 8192) -> str:
    """
    Обрезает контекст до указанного числа токенов
    
    :param tokenizer: токенизатор модели
    :param context: входной текст
    :param max_length: максимальная длина в токенах

    Возвращает обрезанный текст
    """
    tokens = tokenizer.encode(context)
    if len(tokens) > max_length:
        logger.warning(f"Контекст обрезан с {len(tokens)} до {max_length} токенов")
        return tokenizer.decode(tokens[:max_length], skip_special_tokens=True)
    return context


def generate_answer(generator, tokenizer, context: str, question: str) -> str:
    """
    Генерирует ответ на основе контекста и вопроса.
    
    :param generator: пайплайн генерации
    :param tokenizer: токенизатор модели
    :param context: текст из PDF
    :param question: пользовательский запрос
    
    Возвращает сгенерированный ответ
    """

    context = truncate_context(tokenizer, context, max_length=8192)

    prompt = f"""Ты — эксперт по IBM SPSS, помогающий пользователям находить информацию из документации.
Отвечай кратко, ясно и только на основе предоставленного контекста.

Контекст:
{context}

Вопрос:
{question}

1. Не повторяй вопрос.
2. Не добавляй лишней информации.
3. Если ответ найден — дай конкретный ответ.
4. Если ответ не найден — напиши: "Информация не найдена."

Ответ:""".strip()

    try:
        outputs = generator(prompt, max_new_tokens=1024, num_return_sequences=1)
        response = outputs[0]["generated_text"].strip()

        response = response.replace("⁇", "").replace("...", "")

        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response or "Не удалось сформировать ответ."

    except Exception as e:
        logger.error(f"Ошибка при генерации: {e}", exc_info=True)
        return "Ошибка при обработке запроса."