# 🏡✨ House Price Prediction Web App

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

A sleek, intelligent web application built with **Django** that leverages **machine learning** to predict real estate prices in real-time based on user-provided property details.

---

# ✨ Key Features

- 🤖 **Real-Time ML Predictions:** Instantly calculates estimated house prices using a pre-trained machine learning model.  
- 🖥 **Intuitive UI:** A clean, responsive, and user-friendly interface designed for seamless data input.  
- 🧾 **Prediction History:** Automatically stores previous predictions in the database for easy tracking and review.  
- 📊 **Review Dashboard:** Displays a comprehensive log of past predictions directly on the frontend.

---

# 🛠 Technologies Used

| Category | Technologies |
|--------|-------------|
| **Backend Framework** | Django, Python |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **Frontend Styling** | HTML5, CSS3, Bootstrap |

---

# ⚙️ How It Works

1. **Input:** The user enters property details (e.g., area, location, rooms) via the web form.  
2. **Transmission:** The data is securely sent to the Django backend.  
3. **Processing:** The backend processes the input and feeds it into the trained ML model.  
4. **Prediction:** The model calculates the estimated house price.  
5. **Display & Store:** The predicted price is displayed to the user, and the record is saved to the database.

---

# 🚀 Installation & Setup

Follow these steps to get the project running on your local machine.

---

## Prerequisites

Make sure you have a `requirements.txt` file in your root directory containing:

```
Django>=4.0
pandas
numpy
scikit-learn
```

---

## Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

---

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
```

**On Windows**

```bash
venv\Scripts\activate
```

**On Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Django Settings

- Open your `settings.py` file.
- Make sure you use your own unique Django **SECRET_KEY**.

---

### 5️⃣ Train the Machine Learning Model

Before starting the server, you must generate the model files.

```bash
python train_model.py
```

---

### 6️⃣ Verify Model Generation

Check your project directory to see if **two `.pkl` files** have been generated.

- ✅ **IF YES:** Proceed to Step 7.  
- ❌ **IF NO:** Check that all imports are perfectly installed, then re-run the Python file.

---

### 7️⃣ Run the Development Server

```bash
python manage.py runserver
```

---

### 8️⃣ Open the Application

Visit in your browser:

```
http://127.0.0.1:8000/
```

---

# 📂 Project Structure (Example)

```
house-price-predictor/
│
├── predictor/
│   ├── templates/
│   ├── static/
│   ├── views.py
│   ├── models.py
│
├── train_model.py
├── requirements.txt
├── manage.py
└── README.md
```

---

# 📊 Machine Learning Model

The project uses a **Scikit-learn regression model** trained on housing data.  
The trained model is saved as `.pkl` files and loaded by the Django backend for real-time predictions.

---

# 📜 License

This project is licensed under the **MIT License**.

---

# ⭐ Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository  
2. Create a new branch  
3. Commit your changes  
4. Submit a Pull Request

---

# 💡 Future Improvements

- Deploy to **AWS / Render / Railway**
- Add **interactive price charts**
- Improve **ML model accuracy**
- Add **user authentication**

---

# 👨‍💻 Author

Developed with ❤️ using **Django + Machine Learning**.

## Very Important For YOU :
Ideas make projects better.

⚡ This project is only version **today**.  
Your idea could define version **tomorrow**.

Feel free to open an issue, suggest improvements, or submit a pull request. 

Contact through Email ID: umaashankara75189@gmail.com
