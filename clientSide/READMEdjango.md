## 🚀 How to Run the Client-Side Website (Local Setup)

Follow the steps below to run the project on your local machine.

---

### Step 1: Open Command Prompt

- Press **Windows + R**
- Type `cmd`
- Press **Enter**

---

### Step 2: Activate Virtual Environment

Run the following command:


perform_env_312\Scripts\activate


After activation, you should see the environment name in the terminal.

Example:

(perform_env_312)


---

### Step 3: Go to Project Folder

Navigate to the Django project folder:


cd performproject


---

### Step 4: Run the Development Server

Start the Django server on port 8001:


python manage.py runserver 8001


You should see:


Starting development server at http://127.0.0.1:8001/


---

### Step 5: Open in Browser

- Hold **Ctrl**
- Right-click on the URL in terminal  
- Or copy and paste it into your browser:


http://127.0.0.1:8001/


---

## 📌 Notes

- Make sure Python 3.12 is installed.
- Install dependencies if needed:


pip install -r requirements.txt


- Keep the terminal open while the server is running.
- Press **Ctrl + C** to stop the server.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| activate not found | Check virtual environment path |
| Port already in use | Try another port (8002 / 8003) |
| Module error | Run `pip install -r requirements.txt` |

---