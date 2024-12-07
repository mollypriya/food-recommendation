
# Food Recommendation System

This project is a **Personalized Food Recommendation System** designed to recommend meals based on user preferences, nutritional requirements, and allergen restrictions. It uses machine learning algorithms like **K-Nearest Neighbors (KNN)** and **Principal Component Analysis (PCA)** to provide tailored suggestions. The system is implemented using **Flask** for the web interface and various machine learning techniques to process and predict food recommendations.

## Features

- **Personalized Recommendations**: Recommends food items based on user input such as age, weight, height, activity level, and dietary goals.
- **Allergen-Based Filtering**: Filters out foods based on the user’s allergy information.
- **Nutritional Analysis**: Uses TDEE (Total Daily Energy Expenditure) and macronutrient formulas to recommend foods aligned with the user’s goals (e.g., weight loss, balanced diet).
- **Real-Time Adaptability**: The system adapts and provides dynamic recommendations based on changing user input.
- **Interactive Web Interface**: Built with Flask to allow users to interact with the system and view their personalized food recommendations.

## Installation

### Prerequisites

- Python 3.x
- Flask
- scikit-learn
- Pandas
- Numpy
- Matplotlib (for visualizations)
- imbalanced-learn (for SMOTE)

### Steps to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/mollypriya/food-recommendation.git
   ```
   
2. Navigate to the project folder:
   ```bash
   cd food-recommendation
   ```
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open the application in a web browser:
   ```bash
   http://127.0.0.1:5000/
   ```

## How It Works

1. **User Input**: The user enters their details, including age, weight, height, activity level, goal (e.g., weight loss), and allergies.
   
2. **Nutritional Calculation**: The system calculates the user's nutritional requirements using the **TDEE** and **macronutrient formulas**.
   
3. **Recommendation Process**: The KNN algorithm identifies the nearest food items based on the user’s nutritional input, and recommendations are provided. PCA is used to visualize the relationships between food items.

4. **Allergen Filtering**: The system filters out any food items that match the user's allergy profile.

5. **Results**: The user is shown a list of personalized, safe, and nutritionally balanced food items.

## Technologies Used

- **Python**: Core programming language.
- **Flask**: Web framework for developing the user interface.
- **scikit-learn**: Machine learning library for KNN, PCA, and SMOTE.
- **Pandas**: Data manipulation and analysis.
- **Numpy**: Numerical computations.
- **Matplotlib**: Visualizations (for PCA plotting).



## Contact

- **Email**: mollypriya2003@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/molly-priya)


