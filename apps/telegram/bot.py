import os
import telebot
from src.scripts.model_load import predict

TOKEN = os.getenv('TELEGRAM_TOKEN')
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello! I'm a LLM safety guardian bot 🤖")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    response = predict([message.text])
    bot.reply_to(message, response)

bot.infinity_polling()
