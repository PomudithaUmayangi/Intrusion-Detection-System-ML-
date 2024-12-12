# Intrusion Detection System (IDS) - Phishing Detection

## Project Overview

This project focuses on the development of an **Intrusion Detection System (IDS)** designed to detect phishing attacks using machine learning techniques. The system identifies suspicious activities or security breaches within a network, providing a crucial layer of defense against cyber threats. The **KDD Cup 1999 dataset** was used to train the model, which contains various types of attacks and normal data points to build a reliable phishing detection model.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [System Architecture](#system-architecture)
5. [Data Collection](#data-collection)
6. [Machine Learning Model](#machine-learning-model)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

---

## Introduction

The primary goal of this IDS is to enhance the security of online systems by detecting phishing attempts that have become a significant concern in cybersecurity. Using **machine learning**, the system is capable of detecting phishing URLs, identifying suspicious behaviors, and preventing unauthorized access. This project aims to provide real-time phishing attack detection with minimal false positives.

---

## Features

- **Real-time Monitoring:** The system continuously monitors incoming network data for potential phishing threats.
- **Phishing Detection:** Utilizes machine learning models to detect phishing websites and malicious URLs.
- **Alerting System:** Sends real-time notifications when a phishing attack is detected.
- **Reporting:** Detailed reports are generated that include threat severity and timestamps.
- **Scalability:** The system can handle various network traffic volumes and adapt to emerging phishing techniques.

---

## Technologies Used

- **Python:** The programming language used for backend development and model implementation.
- **Scikit-learn:** A machine learning library used for training and evaluating models.
- **Pandas:** For data manipulation and cleaning.
- **Matplotlib/Seaborn:** Used for visualizing the results of the data analysis.
- **Flask/Django:** Optional web framework for real-time monitoring and reporting (if a user interface is needed).
- **MySQL/PostgreSQL:** For storing the logs and detected threats (if applicable).
- **TensorFlow/Keras:** For training more advanced deep learning models (optional).

---

## System Architecture

The system architecture is designed with three key layers:

1. **Data Collection Layer:** Captures network traffic data or user activity logs to identify potential phishing attempts.
2. **Machine Learning Layer:** Processes the collected data, uses pre-trained models, and classifies it to detect phishing.
3. **Alerting and Reporting Layer:** Sends real-time alerts and generates logs/reports for system administrators.

---

## Data Collection

The dataset used for training the machine learning model is the **KDD Cup 1999 dataset**, which is publicly available. This dataset includes:

- Network traffic data and different types of attacks, labeled as either normal or various types of intrusions.
- Phishing features extracted from the data, such as the protocol type, service, flag, and others.

The **KDD Cup 1999** dataset can be accessed from [KDD Cup Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html). This dataset is one of the most commonly used for intrusion detection research.

---

## Machine Learning Model

The machine learning model in this IDS has been trained to recognize phishing attempts and anomalies in network data using features such as:

- URL length
- Number of special characters in the URL
- Presence of phishing-related keywords (e.g., "login," "secure," "account")

We used **Random Forest** and **Support Vector Machines (SVM)** for classification. The model was evaluated on the KDD Cup dataset to ensure its performance in detecting phishing attacks.

---

## Installation

To set up this IDS on your local machine, follow these steps:

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Pip (Python package manager)
- MySQL/PostgreSQL (if using a database)

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/pomudithaumayangi/intrusion-detection-system.git
    cd intrusion-detection-system
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the database (if applicable):**
    - Create a new database and tables.
    - Update the database configuration file with the correct credentials.

4. **Run the system:**

    ```bash
    python run.py
    ```

---

## Usage

Once the system is running:

- **Monitor Network Traffic:** The system will continuously monitor incoming data.
- **View Alerts:** Alerts will be generated if phishing activity is detected.
- **Generate Reports:** Reports will show information on detected threats.

You can view the system's dashboard (if applicable) by navigating to:

```bash
http://localhost:5000
