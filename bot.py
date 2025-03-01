import logging
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# توكن البوت من متغيرات البيئة
TOKEN = os.getenv("7219331966:AAEHT6-IJO7DiAezb1J3sgtbKDrFhJ-kRoE")

# تحميل نموذج الذكاء الاصطناعي (Llama 2 أو Mistral)
model_name = "mistralai/Mistral-7B-Instruct"  # يمكنك استبداله بـ Llama2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# تهيئة البوت
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

logging.basicConfig(level=logging.INFO)

# أمر /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("مرحبًا بك في بوت الإخلاء! 🤍\nأرسل سؤالك وسنرد عليك بالذكاء الاصطناعي.")

# وظيفة الذكاء الاصطناعي
async def get_ai_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# استقبال الرسائل والرد عليها
@dp.message_handler()
async def handle_message(message: types.Message):
    user_text = message.text
    response = await get_ai_response(user_text)
    await message.reply(response)

# تشغيل البوت
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
