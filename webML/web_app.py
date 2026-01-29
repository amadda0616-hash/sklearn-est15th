import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the trained model
# Using joblib as specified in the modeling step
model = joblib.load('model.pkl')

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create a DataFrame for the input with correct column names (matching features used in training)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # Map input values to match training data types if necessary
    # Pclass is int [1, 2, 3]
    # Sex is string ['male', 'female']
    # Embarked is string ['S', 'C', 'Q']
    # Age, SibSp, Parch, Fare are numeric
    
    input_df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=features)
    
    # Predict
    # The pipeline handles preprocessing (imputing, encoding, scaling)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
    
    survival_status = "Survived" if prediction[0] == 1 else "Did Not Survive"
    
    # Optional: Add probability info
    if probability is not None:
        confidence = np.max(probability)
        return f"{survival_status} ({confidence*100:.1f}%)"
    else:
        return survival_status

# Define Gradio Interface
iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Passenger Class (Pclass)", info="1=1st, 2=2nd, 3=3rd"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="Age", value=30, minimum=0, maximum=100),
        gr.Number(label="Siblings/Spouses Aboard (SibSp)", value=0, minimum=0),
        gr.Number(label="Parents/Children Aboard (Parch)", value=0, minimum=0),
        gr.Number(label="Fare", value=32.2, minimum=0),
        gr.Radio(["S", "C", "Q"], label="Port of Embarkation", info="S=Southampton, C=Cherbourg, Q=Queenstown")
    ],
    outputs="text",
    title="Titanic Survivor Prediction Service",
    description="Enter passenger details to predict survival probabilities using an Ensemble Voting Classifier."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
