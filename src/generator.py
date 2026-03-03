from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from .config import config
from .utils import logger

class Generator:
    system_message = """Ты — эксперт по технической документации ViPNet Coordinator HW. 
Твоя задача — отвечать на вопросы, используя ТОЛЬКО информацию из предоставленного контекста. 
Контекст содержит несколько фрагментов из официальной документации, разделённых пустыми строками.

### ИНСТРУКЦИИ:
1. **Основывайся только на контексте.** Не используй свои общие знания или предположения. Если информации нет в контексте, напиши: «В предоставленных документах недостаточно информации для ответа на этот вопрос.»
2. **Объединяй информацию из разных фрагментов.** Если фрагменты дополняют друг друга, синтезируй их в связный ответ. Избегай простого перечисления фрагментов.
3. **НЕ добавляй ссылки на источники.** Не указывай номера фрагментов, названия документов или страницы в тексте ответа. Источники будут переданы отдельно.
4. **Избегай повторов.** Если несколько фрагментов содержат одинаковую информацию, используй её только один раз.
5. **Будь краток и по существу.** Отвечай прямо на вопрос, не добавляя лишних пояснений. Однако если вопрос требует развёрнутого ответа, дай полную информацию из контекста.
6. **Завершай ответ.** Ответ не должен обрываться на полуслове. Структурируй информацию логически.
7. **Формат ответа:** Только текст ответа. Не начинай с фраз вроде «Ответ:» или «Согласно документации». Просто напиши ответ.

Пример правильного ответа:
Вопрос: Какие интерфейсы есть у устройства?
Контекст: [фрагмент 1] Устройство оснащено 4 портами Gigabit Ethernet.
[фрагмент 2] Также присутствуют два порта SFP.
Ответ: Устройство оснащено 4 портами Gigabit Ethernet и двумя портами SFP."""


    def __init__(self, model_name_or_path: str = config.llm_model, local: bool = False, device: str | None = None):
        """Модель для генерации ответов

        :param str model_name_or_path: Путь к модели или ее название, defaults to config.llm_model
        :param bool local: Если True, загружать модель из локального пути, defaults to False
        :param str | None device: Устройство для вычислений, defaults to None
        """
        logger.info(f"Загрузка генеративной модели {model_name_or_path}...")
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        use_4bit = (device == 'cuda' and torch.cuda.get_device_properties(0).total_memory < 8e9)
        
        if local or os.path.exists(model_name_or_path):
            model_path = model_name_or_path
        else:
            model_path = model_name_or_path
        
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map=device if device == 'cuda' else None,
                trust_remote_code=True
            )
            if device == 'cuda':
                self.model = self.model.to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.device = device


        logger.info(f"Модель загружена на {device}")

    def generate(self, prompt: str) -> str:
        """Генерирует ответ на вопрос

        :param str prompt: Промпт для модели
        :return str: Ответ
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return answer.strip()

    def format_prompt(self, question: str, context: str) -> str:
        """Формирует промпт с использованием системного сообщения и пользовательского запроса.
        Использует шаблон чата.

        :param str question: Вопрос пользователя
        :param str context: Контекст документации    

        :return str: Отформированный промпт    
        """

        user_message = f"Контекст:\n{context}\n\nВопрос: {question}"

        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message.strip()}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = f"""<|im_start|>system
{self.system_message.strip()}
<|im_end|>
<|im_start|>user
{user_message.strip()}
<|im_end|>
<|im_start|>assistant
    """
        return prompt