import telebot
import cv2
import time

# Initialize the Telegram bot
bot = telebot.TeleBot("7687003019:AAHJhM9gPQLU6eOTBzjhFcHnVaKwK4dWOSQ")
CHAT_ID = "-4515526204"
last_alert_time = 0

def send_telegram_alert(frame, message, priority="medium"):
    print("Telegram alert started")
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time >= 60:
        try:
            cv2.imwrite("alert.jpg", frame)
            with open("alert.jpg", 'rb') as photo:
                # Customize the caption based on priority
                if priority == "high":
                    caption = f"🚨 *URGENT ALERT!* 🚨\n{message}\n⚠️ Take immediate action! ⚠️"
                elif priority == "medium":
                    caption = f"⚠️ *Alert:* ⚠️\n{message}\n🔍 Please investigate."
                else:  # Low priority
                    caption = f"ℹ️ *Information:* ℹ️\n{message}\n✅ All is well."

                # Send the alert with markdown formatting
                bot.send_photo(CHAT_ID, photo, caption=caption, parse_mode='Markdown')

            last_alert_time = current_time
            print(f"Telegram alert sent: {message}")
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
    else:
        print("Waiting to send next alert. Time since last alert:", int(current_time - last_alert_time), "seconds")
