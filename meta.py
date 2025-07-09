import telebot
import os
import cv2
import numpy as np
from PIL import Image
import random
import string
import requests
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from functools import wraps
from collections import defaultdict
import sys
import tempfile
from threading import Lock
import imagehash

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configuration
TOKEN = os.getenv('TOKEN', '7572796786:AAHowjwbLsFRbu0xxZ0ux8fQasKs40DXC3I')
WEBHOOK_URL = 'https://telegram-bot-iypa.onrender.com/webhook'
HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', 10000))
ADMIN_USER_ID = 1019470653

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60
user_requests = defaultdict(list)

# Initialize Flask for webhooks
app = Flask(__name__)

# Database lock for thread safety
db_lock = Lock()

# Check Telegram API connection
def check_telegram_api():
    for attempt in range(3):
        try:
            response = requests.get(f'https://api.telegram.org/bot{TOKEN}/getMe', timeout=10)
            if response.status_code == 200:
                logging.info("Successfully connected to Telegram API!")
                return True
            else:
                logging.error(f"Connection failed: Status code {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Telegram API connection error (attempt {attempt+1}/3): {str(e)}")
            if attempt < 2:
                time.sleep(5)
    logging.error("Failed to connect to Telegram API after retries.")
    return False

# Initialize bot with retries
def init_bot(retries=3, delay=10):
    for attempt in range(retries):
        try:
            bot = telebot.TeleBot(TOKEN, parse_mode=None)
            return bot
        except Exception as e:
            logging.error(f"Bot initialization error (attempt {attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
    logging.error("Failed to initialize bot after retries. Exiting.")
    exit(1)

bot = init_bot()

# Initialize database
def init_db():
    with db_lock:
        try:
            with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    free_photos INTEGER DEFAULT 15,
                    subscription_end TEXT,
                    access_code TEXT
                )''')
                c.execute('''CREATE TABLE IF NOT EXISTS codes (
                    code TEXT PRIMARY KEY,
                    is_used INTEGER DEFAULT 0
                )''')
                conn.commit()
                logging.info("Database initialized successfully.")
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {str(e)}")
            exit(1)

# Generate 50 unique access codes
def generate_access_codes():
    with db_lock:
        try:
            with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                c = conn.cursor()
                c.execute('SELECT COUNT(*) FROM codes')
                if c.fetchone()[0] == 0:
                    for _ in range(50):
                        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                        while True:
                            c.execute('SELECT code FROM codes WHERE code = ?', (code,))
                            if not c.fetchone():
                                break
                            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                        c.execute('INSERT INTO codes (code, is_used) VALUES (?, ?)', (code, 0))
                    conn.commit()
                    logging.info("Generated 50 access codes.")
        except sqlite3.Error as e:
            logging.error(f"Error generating access codes: {str(e)}")

# Generate a specified number of unique codes
def generate_new_codes(num_codes):
    new_codes = []
    with db_lock:
        try:
            with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                c = conn.cursor()
                for _ in range(num_codes):
                    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                    while True:
                        c.execute('SELECT code FROM codes WHERE code = ?', (code,))
                        if not c.fetchone():
                            break
                        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                    c.execute('INSERT INTO codes (code, is_used) VALUES (?, ?)', (code, 0))
                    new_codes.append(code)
                conn.commit()
                logging.info(f"Generated {num_codes} new codes.")
            return new_codes
        except sqlite3.Error as e:
            logging.error(f"Error generating new codes: {str(e)}")
            return []

# Load Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    logging.error("Error: Could not load haarcascade_frontalface_default.xml")
    exit(1)

# Generate random filename in temp directory
def generate_random_filename(prefix=""):
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    filename = os.path.join(tempfile.gettempdir(), f"{prefix}{random_string}.jpg")
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(tempfile.gettempdir(), f"{prefix}{random_string}_{counter}.jpg")
        counter += 1
    return filename

# Remove metadata from image
def remove_metadata(input_path, output_path):
    for attempt in range(3):
        try:
            with Image.open(input_path) as img:
                img = img.convert('RGB')
                img.save(output_path, exif=b'')
            logging.info(f"Metadata removed: {output_path}")
            return True
        except Exception as e:
            logging.warning(f"Metadata removal error (attempt {attempt+1}/3): {str(e)}")
            if attempt < 2:
                time.sleep(1)
    logging.error(f"Failed to remove metadata from {input_path}")
    return False

# Retry file deletion
def safe_remove(filename, retries=5, delay=0.5):
    if not filename:
        return True
    for attempt in range(retries):
        try:
            if os.path.exists(filename):
                os.remove(filename)
                logging.info(f"Deleted file: {filename}")
            return True
        except OSError as e:
            logging.warning(f"Cannot delete {filename} (attempt {attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
    logging.error(f"Failed to delete {filename} after {retries} attempts")
    return False

# Compute perceptual hashes (pHash, dHash, aHash)
def compute_image_hashes(image_path):
    try:
        with Image.open(image_path) as img:
            phash = str(imagehash.phash(img, hash_size=8))
            dhash = str(imagehash.dhash(img, hash_size=8))
            ahash = str(imagehash.average_hash(img, hash_size=8))
        return {'phash': phash, 'dhash': dhash, 'ahash': ahash}
    except Exception as e:
        logging.error(f"Error computing image hashes for {image_path}: {str(e)}")
        return None

# Process image with enhanced transformations
def process_image(image_path, max_size_mb=10):
    logging.info(f"Processing image: {image_path}")
    temp_filename = None
    try:
        if os.path.getsize(image_path) > max_size_mb * 1024 * 1024:
            logging.error("Image file size exceeds 10MB limit")
            raise Exception("Image file size exceeds 10MB limit")
        temp_filename = generate_random_filename(prefix="temp_")
        if not remove_metadata(image_path, temp_filename):
            raise Exception("Failed to remove metadata")

        # Compute hashes for input image
        input_hashes = compute_image_hashes(temp_filename)
        if input_hashes:
            logging.info(f"Input image hashes: pHash={input_hashes['phash']}, dHash={input_hashes['dhash']}, aHash={input_hashes['ahash']}")
        else:
            logging.warning("Failed to compute input image hashes")

        image = cv2.imread(temp_filename)
        if image is None:
            logging.error("Failed to load image with OpenCV")
            raise Exception("Failed to load image with OpenCV")

        # Apply global transformations for hash differences
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        angle = random.uniform(-15, 15)  # Stronger rotation
        scale = random.uniform(0.9, 1.1)  # Random scaling
        M = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        logging.info(f"Applied global rotation: {angle} degrees, scale: {scale}")

        # Apply color shift
        color_shift = np.random.randint(-20, 21, size=(3,))
        image = image.astype(np.float32)
        for channel in range(3):
            image[:, :, channel] += color_shift[channel]
        image = np.clip(image, 0, 255).astype(np.uint8)
        logging.info(f"Applied color shift: {color_shift}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        logging.info(f"Detected {len(faces)} faces")

        if len(faces) > 0:
            blur_kernel = 11  # Stronger blur
            blur_sigma = 4.0  # Stronger sigma
            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(face_roi, (blur_kernel, blur_kernel), blur_sigma)
                image[y:y+h, x:x+w] = blurred_face
                rows, cols = face_roi.shape[:2]
                M = np.float32([[1, 0.03, 6], [0.03, 1, 6]])  # Stronger shear
                face_roi = cv2.warpAffine(face_roi, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
                image[y:y+h, x:x+w] = face_roi

        alpha = 1.2  # Stronger contrast
        beta = 15  # Stronger brightness
        noise_percentage = 0.02  # Stronger noise
        noise_strength = 10  # Stronger noise strength
        shuffle_percentage = 0.01  # Stronger pixel shuffling
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        height, width = image.shape[:2]
        for _ in range(int(height * width * noise_percentage)):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            for channel in range(3):
                pixel_value = int(image[y, x, channel])
                delta = random.randint(-noise_strength, noise_strength)
                new_value = max(0, min(255, pixel_value + delta))
                image[y, x, channel] = new_value
        for _ in range(int(height * width * shuffle_percentage)):
            x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
            x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
            image[y1, x1], image[y2, x2] = image[y2, x2], image[y1, x1]
        noise = np.random.normal(0, 1.5, image.shape).astype(np.float32)  # Stronger Gaussian noise
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        output_filename = generate_random_filename(prefix="output_")
        success = cv2.imwrite(output_filename, image)
        if not success or not os.path.exists(output_filename):
            logging.error("Image processing failed: output file not created.")
            raise Exception("Image processing failed.")

        # Compute hashes for output image
        output_hashes = compute_image_hashes(output_filename)
        if output_hashes:
            logging.info(f"Output image hashes: pHash={output_hashes['phash']}, dHash={output_hashes['dhash']}, aHash={output_hashes['ahash']}")
        else:
            logging.warning("Failed to compute output image hashes")

        logging.info(f"Processed image saved: {output_filename}")
        return output_filename
    except Exception as e:
        logging.error(f"Image processing error: {str(e)}")
        raise
    finally:
        safe_remove(temp_filename)

# Check user status
def check_user_status(user_id):
    with db_lock:
        try:
            with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                c = conn.cursor()
                c.execute('SELECT free_photos, subscription_end, access_code FROM users WHERE user_id = ?', (user_id,))
                user = c.fetchone()
                if not user:
                    c.execute('INSERT INTO users (user_id, free_photos) VALUES (?, ?)', (user_id, 15))
                    conn.commit()
                    return 'free', 15
                free_photos, subscription_end, access_code = user
                if free_photos > 0:
                    return 'free', free_photos
                if subscription_end:
                    end_date = datetime.fromisoformat(subscription_end)
                    if datetime.now() < end_date:
                        return 'subscribed', None
                    else:
                        c.execute('UPDATE users SET subscription_end = NULL, access_code = NULL WHERE user_id = ?', (user_id,))
                        conn.commit()
                return 'none', None
        except sqlite3.Error as e:
            logging.error(f"Error checking user status: {str(e)}")
            return 'none', None

# Rate limiting decorator
def rate_limit(f):
    @wraps(f)
    def decorated_function(message):
        user_id = message.from_user.id
        current_time = time.time()
        user_requests[user_id] = [t for t in user_requests[user_id] if current_time - t < RATE_LIMIT_WINDOW]
        if len(user_requests[user_id]) >= RATE_LIMIT_REQUESTS:
            bot.reply_to(message, "Too many requests. Please try again later.")
            return
        user_requests[user_id].append(current_time)
        return f(message)
    return decorated_function

# Admin check decorator
def admin_only(f):
    @wraps(f)
    def decorated_function(message):
        if message.from_user.id != ADMIN_USER_ID:
            bot.reply_to(message, "You are not authorized to use this command.")
            return
        return f(message)
    return decorated_function

# Start command handler
@bot.message_handler(commands=['start'])
@rate_limit
def send_welcome(message):
    welcome_message = (
        "Hey there! 👋\n"
        "This is the most powerful Image Spoofer for all major dating apps.\n"
        "You have 15 free tries to test it out.\n\n"
        "After that, contact @SourceCode101 to purchase your license."
    )
    bot.reply_to(message, welcome_message)

# Redeem access code
@bot.message_handler(commands=['redeem'])
@rate_limit
def redeem_code(message):
    user_id = message.from_user.id
    status, _ = check_user_status(user_id)
    if status == 'subscribed':
        bot.reply_to(message, "You already have an active subscription!")
        return
    bot.reply_to(message, "Enter your access code:")
    bot.register_next_step_handler(message, process_code)

def process_code(message):
    user_id = message.from_user.id
    code = message.text.strip()
    with db_lock:
        try:
            with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                c = conn.cursor()
                c.execute('SELECT is_used FROM codes WHERE code = ?', (code,))
                result = c.fetchone()
                if result and result[0] == 0:
                    c.execute('UPDATE codes SET is_used = 1 WHERE code = ?', (code,))
                    subscription_end = (datetime.now() + timedelta(days=30)).isoformat()
                    c.execute('UPDATE users SET access_code = ?, subscription_end = ? WHERE user_id = ?',
                              (code, subscription_end, user_id))
                    conn.commit()
                    bot.reply_to(message, "Code activated! You have access for 30 days.")
                else:
                    bot.reply_to(message, "Invalid or already used code!")
        except sqlite3.Error as e:
            logging.error(f"Error processing code: {str(e)}")
            bot.reply_to(message, "Error processing code. Try again later.")

# Admin commands for code management
@bot.message_handler(commands=['addcode'])
@rate_limit
@admin_only
def add_code(message):
    try:
        code = message.text.split()[1].strip()
        with db_lock:
            with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                c = conn.cursor()
                c.execute('SELECT code FROM codes WHERE code = ?', (code,))
                if c.fetchone():
                    bot.reply_to(message, "This code already exists!")
                    return
                c.execute('INSERT INTO codes (code, is_used) VALUES (?, ?)', (code, 0))
                conn.commit()
                bot.reply_to(message, f"Code {code} added successfully!")
    except IndexError:
        bot.reply_to(message, "Please provide a code: /addcode <code>")
    except Exception as e:
        logging.error(f"Error adding code: {str(e)}")
        bot.reply_to(message, f"Error adding code: {str(e)}")

@bot.message_handler(commands=['removecode'])
@rate_limit
@admin_only
def remove_code(message):
    try:
        code = message.text.split()[1].strip()
        with db_lock:
            with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                c = conn.cursor()
                c.execute('DELETE FROM codes WHERE code = ?', (code,))
                if conn.total_changes > 0:
                    conn.commit()
                    bot.reply_to(message, f"Code {code} removed successfully!")
                else:
                    bot.reply_to(message, "Code not found!")
    except IndexError:
        bot.reply_to(message, "Please provide a code: /removecode <code>")
    except Exception as e:
        logging.error(f"Error removing code: {str(e)}")
        bot.reply_to(message, f"Error removing code: {str(e)}")

@bot.message_handler(commands=['listcodes'])
@rate_limit
@admin_only
def list_codes(message):
    with db_lock:
        try:
            with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                c = conn.cursor()
                c.execute('SELECT code, is_used FROM codes')
                codes = c.fetchall()
                if not codes:
                    bot.reply_to(message, "No codes available.")
                    return
                response = "Available codes:\n"
                for code, is_used in codes:
                    status = "Used" if is_used else "Unused"
                    response += f"{code} - {status}\n"
                bot.reply_to(message, response)
        except sqlite3.Error as e:
            logging.error(f"Error listing codes: {str(e)}")
            bot.reply_to(message, "Error listing codes. Try again later.")

@bot.message_handler(commands=['generatecodes'])
@rate_limit
@admin_only
def generate_codes(message):
    try:
        num_codes = int(message.text.split()[1])
        if num_codes <= 0 or num_codes > 100:
            bot.reply_to(message, "Please specify a number between 1 and 100.")
            return
        new_codes = generate_new_codes(num_codes)
        response = f"Generated {num_codes} new codes:\n" + "\n".join(new_codes)
        bot.reply_to(message, response)
    except IndexError:
        bot.reply_to(message, "Please provide a number: /generatecodes <number>")
    except ValueError:
        bot.reply_to(message, "Please provide a valid number: /generatecodes <number>")
    except Exception as e:
        logging.error(f"Error generating codes: {str(e)}")
        bot.reply_to(message, f"Error generating codes: {str(e)}")

# Handle photo
@bot.message_handler(content_types=['photo'])
@rate_limit
def handle_photo(message):
    user_id = message.from_user.id
    logging.info(f"Received photo from user {user_id}")
    status, free_photos = check_user_status(user_id)
    
    if status == 'none':
        bot.reply_to(message, "You've used all your free photos! To continue, DM @SourceCode101 to purchase an access code.")
        return
    
    input_filename = None
    output_filename = None
    try:
        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        if not file_info.file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            bot.reply_to(message, "Unsupported image format. Please send a JPG or PNG image.")
            return
        logging.info(f"Downloading photo: {file_info.file_path}")
        downloaded_file = bot.download_file(file_info.file_path)
        input_filename = generate_random_filename(prefix="input_")
        with open(input_filename, 'wb') as new_file:
            new_file.write(downloaded_file)
        logging.info(f"Saved input photo: {input_filename}")

        output_filename = process_image(input_filename)
        logging.info(f"Image processed: {output_filename}")
        if not os.path.exists(output_filename):
            logging.error(f"Processed file {output_filename} does not exist")
            raise Exception("Processed image file not found")
        with open(output_filename, 'rb') as photo:
            bot.send_photo(message.chat.id, photo=photo)
            logging.info(f"Sent processed photo: {output_filename}")
        
        if status == 'free':
            with db_lock:
                try:
                    with sqlite3.connect('/data/bot.db', check_same_thread=False) as conn:
                        c = conn.cursor()
                        c.execute('UPDATE users SET free_photos = free_photos - 1 WHERE user_id = ?', (user_id,))
                        conn.commit()
                    bot.reply_to(message, f"Photo processed! {free_photos - 1} free photos remaining.")
                except sqlite3.Error as e:
                    logging.error(f"Error updating user photos: {str(e)}")
                    bot.reply_to(message, "Error updating photo count. Try again later.")
    except Exception as e:
        logging.error(f"Error processing photo: {str(e)}")
        bot.reply_to(message, f"Error: {str(e)}")
    finally:
        safe_remove(input_filename)
        safe_remove(output_filename)

# Webhook endpoint
@app.route('/webhook', methods=['POST'])
def webhook():
    logging.debug("Received webhook request")
    try:
        update = telebot.types.Update.de_json(request.get_json())
        if update:
            logging.info(f"Processing update: {update.update_id}")
            bot.process_new_updates([update])
        else:
            logging.warning("Received empty or invalid update")
        return jsonify({"status": "ok"})
    except Exception as e:
        logging.error(f"Webhook error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Start bot
def start_bot():
    init_db()
    generate_access_codes()
    if not check_telegram_api():
        logging.error("Failed to connect to Telegram API. Falling back to polling.")
        bot.polling(none_stop=True)
        return
    try:
        logging.info("Deleting any existing webhook...")
        bot.delete_webhook()
        time.sleep(1)
        logging.info(f"Setting webhook to {WEBHOOK_URL}...")
        bot.set_webhook(url=WEBHOOK_URL)
        logging.info(f"Webhook set to {WEBHOOK_URL}")
        app.run(host=HOST, port=PORT)
    except Exception as e:
        logging.error(f"Failed to set webhook: {str(e)}. Falling back to polling.")
        bot.delete_webhook()
        bot.polling(none_stop=True)

if __name__ == "__main__":
    start_bot()