from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
import threading

app = Flask(__name__)

# Dictionary to store WebDriver instances and chat numbers
sessions = {}

# Function to create a new WebDriver instance
def create_session(ip):
    # Set up Chrome options
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

    # Initialize WebDriver
    driver = webdriver.Chrome(options=options)
    # Navigate to the target URL
    driver.get(os.get(HelpingAI))
    time.sleep(7)

    # Security Bypass: Refresh the page if the title contains 'just a moment'
    while 'just a moment' in driver.title.lower():
        driver.refresh()

    # Initialize Chat_Num
    Chat_Num = 2

    sessions[ip] = {"driver": driver, "Chat_Num": Chat_Num, "last_active": time.time()}

# Function to close inactive sessions
def close_inactive_sessions():
    while True:
        for ip, session in list(sessions.items()):
            if time.time() - session["last_active"] > 180:  # 3 minutes
                session["driver"].quit()
                del sessions[ip]
        time.sleep(60)  # Check every minute

# Start a background thread to close inactive sessions
threading.Thread(target=close_inactive_sessions, daemon=True).start()

# API route to handle queries
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    ip = request.remote_addr  # Get the IP address of the client
    query = data['query']

    # If there's no session for this IP, create one
    if ip not in sessions:
        create_session(ip)

    session = sessions[ip]
    driver = session["driver"]
    Chat_Num = session["Chat_Num"]

    # Function to increment Chat_Num
    def increment_chat_num():
        nonlocal Chat_Num
        Chat_NumNew = int(Chat_Num) + 1
        Chat_NumNew = str(Chat_NumNew)
        Chat_Num = Chat_NumNew
        session["Chat_Num"] = Chat_Num

    # Function to send a query and retrieve response
    def send_query(query):
        text_box_xpath = "/html/body/div[1]/main/div[1]/div/div/div/div/div/div/div/form/fieldset/textarea"
        send_button_xpath = "/html/body/div[1]/main/div[1]/div/div/div/div/div/div/div/form/fieldset/button"
        response_xpath = f"/html/body/div[1]/main/div[1]/div/div/div/div/div/div/div/div/div/div[{Chat_Num}]/div[2]"
        button_xpath = f"/html/body/div[1]/main/div[1]/div/div/div/div/div/div/div/div/div/div[{Chat_Num}]/div[1]/div/form/div/div[1]/button"

        # Find the text box, enter query, and click send
        text_box = driver.find_element(by=By.XPATH, value=text_box_xpath)
        text_box.clear()
        text_box.send_keys(query)
        time.sleep(0.25)  # Pause for 1 second after typing query

        send_button = driver.find_element(by=By.XPATH, value=send_button_xpath)
        send_button.click()

        # Continuously check for the presence of the button every second
        while True:
            try:
                button = driver.find_element(by=By.XPATH, value=button_xpath)
                # If the button is found, retrieve and return the response
                response = driver.find_element(by=By.XPATH, value=response_xpath).text
                return response
            except NoSuchElementException:
                time.sleep(
                    0.25
                )  # If the button is not found, wait for 1 second before checking again

    response = send_query(query)
    increment_chat_num()
    session["last_active"] = time.time()  # Update the last active time
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
