# from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer

# app = Flask(__name__)

# file_path = 'foodrecmergedallergen.xlsx'
# data = pd.read_excel(file_path)

# # Normalize Numerical Features
# scaler = StandardScaler()
# X_numerical = scaler.fit_transform(data[['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Carbohydrates', 'Fibre', 'VitaminD', 'Sugars']])

# # Train KNN Model
# knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
# knn.fit(X_numerical)

# def recommend_recipes(input_features, user_allergy):
#     # Scale the numerical features
#     input_features_scaled = scaler.transform([input_features])
    
#     # Get KNN recommendations
#     distances, indices = knn.kneighbors(input_features_scaled)
#     recommendations = data.iloc[indices[0]]

#     # Filter out recommendations based on the user's allergy
#     filtered_recommendations = recommendations[~recommendations['Food_allergy'].str.contains(user_allergy, case=False, na=False)]

#     # Return the filtered food items along with allergy info
#     return filtered_recommendations[['Food_items','image_url']]
# # Absolute path example
# # file_path = 'foodrecmergedallergen.xlsx'
# # data = pd.read_excel(file_path)
# # #data = pd.read_csv("foodrecmergedallergen.xlsx")


# # # Preprocess Ingredients
# # vectorizer = TfidfVectorizer()
# # X_ingredients = vectorizer.fit_transform(data['Food_items'])
# # #vectorizer = TfidfVectorizer()
# # #X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# # # Normalize Numerical Features
# # scaler = StandardScaler()
# # X_numerical = scaler.fit_transform(data[['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Carbohydrates','Fibre','VitaminD','Sugars']])
# # #X_numerical = scaler.fit_transform(recipe_df[['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium']])

# # # Normalize Numerical Features
# # #scaler = StandardScaler()
# # #X_numerical = scaler.fit_transform(data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

# # # Combine Features
# # X_combined = np.hstack([X_numerical, X_ingredients.toarray()])
# # #X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# # # Train KNN Model
# # knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
# # knn.fit(X_combined)
# # # Combine Features
# # X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# # # Train KNN Model
# # knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
# # knn.fit(X_combined)

# # def recommend_recipes(input_features):
# #     input_features_scaled = scaler.transform([input_features[:7]])
# #     input_ingredients_transformed = vectorizer.transform([input_features[7]])
# #     input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
# #     distances, indices = knn.kneighbors(input_combined)
# #     recommendations = data.iloc[indices[0]]
# #     return recommendations[['recipe_name', 'ingredients_list', 'image_url']].head(5)

# # Function to truncate product name
# def truncate(text, length):
#     if len(text) > length:
#         return text[:length] + "..."
#     else:
#         return text

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         calories = float(request.form['calories'])
#         fat = float(request.form['fat'])
#         protein = float(request.form['protein'])
#         iron = float(request.form['iron'])
#         calcium = float(request.form['calcium'])
#         sodium = float(request.form['sodium'])
#         potassium=float(sodium = float(request.form['potassium']))
#         carbohydrates = float(request.form['carbohydrates'])
#         fiber = float(request.form['fiber'])
#         vitamind= float(request.form['vitamind'])
#         sugars=float(request.form['sugars'])
        
        
#         #ingredients = request.form['ingredients']
#         user_allergy = request.form['userallergy']
#         input_features = [calories, fat, protein, iron,calcium, sodium,potassium,carbohydrates, fiber,vitamind,sugars]
#         recommendations = recommend_recipes(input_features,user_allergy)
#         return render_template('index.html', recommendations=recommendations.to_dict(orient='records'),truncate = truncate)
#     return render_template('index.html', recommendations=[])

# if __name__ == '__main__':
#     app.run(debug=True)
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE
# from sklearn.decomposition import PCA
# from flask import Flask, render_template, request
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)
# # Load the dataset
# file_path = 'foodrecmergedallergen.xlsx'
# recipe_df = pd.read_excel(file_path)

# # Example User Allergy List
# # user_allergies = ['Gluten']  # Replace with actual user allergies

# # Fill NaN values in 'Food_allergy' with 'Unknown'
# recipe_df['Food_allergy'] = recipe_df['Food_allergy'].fillna('Unknown')


# # Remove classes with very few samples (e.g., less than 5)
# min_class_count = 5  # Set the threshold for the minimum number of samples per class
# class_counts = recipe_df['Food_allergy'].value_counts()
# minority_classes = class_counts[class_counts < min_class_count].index
# recipe_df = recipe_df[~recipe_df['Food_allergy'].isin(minority_classes)]


# # Feature extraction for ingredients and nutritional values
# vectorizer = TfidfVectorizer()
# X_ingredients = vectorizer.fit_transform(recipe_df['Food_items'])

# # Normalize Numerical Features
# scaler = StandardScaler()
# X_numerical = scaler.fit_transform(recipe_df[['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Carbohydrates', 'Fibre', 'VitaminD', 'Sugars']])

# # Combine Features
# X_combined = np.hstack([X_numerical, X_ingredients.toarray()])
# y = recipe_df['Food_allergy']

# # Apply SMOTE for class balancing (now should work as we've removed small classes)
# smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
# X_resampled, y_resampled = smote.fit_resample(X_combined, y)

# # Apply PCA for dimensionality reduction (optional but can improve model performance)
# pca = PCA(n_components=0.95)  # Keep 95% of the variance
# X_resampled = pca.fit_transform(X_resampled)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# # Train KNN Classifier Model with optimized hyperparameters
# knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
# knn_classifier.fit(X_train, y_train)

# # Predict on the test set
# knn_predictions = knn_classifier.predict(X_test)



# def recommend_recipes(input_features, user_allergy, scaler, vectorizer, knn_classifier, pca, recipe_df):
#     # Step 1: Extract and reshape the input nutritional data
#     nutritional_input = np.array(input_features).reshape(1, -1)  # Nutritional features (1 sample)
    
#     # Step 2: Scale the nutritional features using the scaler (apply same scaling as during training)
#     nutritional_input_scaled = scaler.transform(nutritional_input)
    
#     # Step 3: Combine with TF-IDF vector (assuming no ingredients are provided in input)
#     ingredients_input = vectorizer.transform(['']).toarray()  # Placeholder for ingredients (empty string or dummy)
    
#     # Combine nutritional input and ingredient input
#     combined_input = np.hstack([nutritional_input_scaled, ingredients_input])
    
#     # Step 4: Apply PCA transformation to match the PCA space used during training
#     combined_input_pca = pca.transform(combined_input)
    
#     # Step 5: Ensure the PCA-transformed input is within valid range for KNN
#     # Here, combined_input_pca is a 1x33 array, corresponding to the number of components in PCA
#     # Now we can safely get KNN indices without worrying about out-of-bounds errors
    
#     distances, indices = knn_classifier.kneighbors(combined_input_pca)
    
#     # Debug: Check the indices returned by KNN
#     # print(f"KNN indices: {indices}")
    
#     # Step 6: Ensure indices are within bounds of the recipe DataFrame
#     valid_indices = [i for i in indices[0] if i < len(recipe_df)]  # Only keep valid indices within bounds
    
#     if not valid_indices:
#         # print(f"Error: No valid indices found. Max index is {indices[0].max()}, but recipe_df has {len(recipe_df)} rows.")
#         return None
    
#     # Filter the recommendations using valid indices
#     recommendations = recipe_df.iloc[valid_indices]

#     # Step 7: Filter out recipes based on the user's allergy
#     filtered_recommendations = recommendations[~recommendations['Food_allergy'].str.contains(user_allergy, case=False, na=False)]

#     # Step 8: Return the filtered food items along with allergy information
#     #return filtered_recommendations[['Food_items', 'Food_allergy']]
#     return filtered_recommendations[['Food_items', 'image_url', 'Food_allergy']].to_dict(orient='records')


from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE
# from sklearn.decomposition import PCA
from flask import Flask, render_template, request


app = Flask(__name__)

# Load the dataset
file_path = 'foodrecmergedallergen.xlsx'
data = pd.read_excel(file_path)

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['Calories', 'Fats', 'Proteins', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Carbohydrates', 'Fibre', 'VitaminD', 'Sugars']])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=11, metric='euclidean')
knn.fit(X_numerical)

def recommend_recipes(input_features, user_allergy):
    # Scale the numerical features
    input_features_scaled = scaler.transform([input_features])
    
    # Get KNN recommendations
    distances, indices = knn.kneighbors(input_features_scaled)
    recommendations = data.iloc[indices[0]]

    # Filter out recommendations based on the user's allergy
    if user_allergy=='None':
        return recommendations[['Food_items', 'image_url', 'Food_allergy']].to_dict(orient='records')
    else:
        filtered_recommendations = recommendations[~recommendations['Food_allergy'].str.contains(user_allergy, case=False, na=False)]

    # Return the filtered food items along with allergy info
    return filtered_recommendations[['Food_items', 'image_url', 'Food_allergy']].to_dict(orient='records')


# Function to truncate product name
def truncate(text, length=100):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def calculate_bmr(weight, height, age, gender):
    if gender == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr, activity_level):
    activity_factors = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "super_active": 1.9
    }
    return bmr * activity_factors.get(activity_level, 1.2)

def calculate_macronutrients(tdee, goal):
    if goal == "balanced":
        carb_ratio, protein_ratio, fat_ratio = 0.55, 0.2, 0.25
    elif goal == "weight_loss":
        carb_ratio, protein_ratio, fat_ratio = 0.4, 0.3, 0.3
    elif goal == "muscle_gain":
        carb_ratio, protein_ratio, fat_ratio = 0.5, 0.25, 0.25
    else:
        carb_ratio, protein_ratio, fat_ratio = 0.5, 0.2, 0.3  # Default balanced

    carbs = (tdee * carb_ratio) / 4  # Carbohydrates: 4 kcal per gram
    protein = (tdee * protein_ratio) / 4  # Proteins: 4 kcal per gram
    fats = (tdee * fat_ratio) / 9  # Fats: 9 kcal per gram

    return {
        "Calories": tdee,
        "Carbohydrates": carbs,
        "Proteins": protein,
        "Fats": fats
    }


def calculate_micronutrients(age, gender, bmi, activity_level, height, weight):
    # Activity level factor, varies by sedentary, moderate, active
    activity_level_factor = 1 if activity_level == "sedentary" else (1.5 if activity_level == "moderate" else 2)

    # Adjusted values within dataset ranges, based on height, weight, BMI, age, gender, and activity level
    micronutrients = {
        # Iron: adjusted for weight, with different values for gender and age
        "Iron": min(max(0.1 * weight, 0), 15),
        
        # Calcium: scaled by age and weight, fitting the range
        "Calcium": min(max(50 + (0.2 * weight), 50), 100),
        
        # Sodium: varies with activity level and weight, within dataset range
        "Sodium": min(max(300 + (activity_level_factor * 50) + (0.3 * weight), 300), 700),
        
        # Potassium: varies by height and BMI, kept within 200-400 mg
        "Potassium": min(max(200 + (bmi < 25) * (35 * height) or (40 * height), 200), 400),
        
        # Fiber: scales slightly with BMI, stays in 2-3 g range
        "Fibre": min(max(2 + (0.1 * (bmi - 25)) if bmi > 25 else 2, 2), 3),
        
        # Vitamin D: higher for older adults, fitting range
        "VitaminD": 50 if age > 50 else 20,
        
        # Sugars: lower for BMI >= 25, fitting 5–7 g range
        "Sugars": 5 if bmi >= 25 else 7
    }
    return micronutrients



def get_nutritional_requirements(age, weight, height, gender, activity_level, goal):
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)
    macros = calculate_macronutrients(tdee, goal)
    micros = calculate_micronutrients(age,gender,bmr,activity_level,height,weight)

    # Combine both macro and micronutrients
    nutrition_requirements = {**macros, **micros}
    return nutrition_requirements


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        activity_level = request.form['activitylevel']
        height= float(request.form['height'])
        goal = request.form['goal']
        gender=request.form['gender']
        nutrition_needs = get_nutritional_requirements(
            age,
            weight,
            height,
            gender,
            activity_level,
            goal
        )
        # User allergy input
        user_allergy = request.form['userallergy']
        calories=nutrition_needs["Calories"]
        fat=nutrition_needs["Fats"]
        protein=nutrition_needs["Proteins"]
        iron=nutrition_needs["Iron"]
        calcium=nutrition_needs["Calcium"]
        sodium=nutrition_needs["Sodium"]
        potassium=nutrition_needs["Potassium"]
        carbohydrates=nutrition_needs["Carbohydrates"]
        fiber=nutrition_needs["Fibre"]
        vitamind=nutrition_needs["VitaminD"]
        sugars=nutrition_needs["Sugars"]
        
        # Create input feature array
        input_features = [calories, fat, protein, iron, calcium, sodium, potassium, carbohydrates, fiber, vitamind, sugars]
        
        # Get recipe recommendations based on user input
        recommendations = recommend_recipes(input_features, user_allergy)
        #recommendations = recommend_recipes(input_features, user_allergy, scaler, vectorizer, knn_classifier, pca, recipe_df)

        return render_template('index.html', recommendations=recommendations, truncate=truncate)
    
    return render_template('index.html', recommendations=[])

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

