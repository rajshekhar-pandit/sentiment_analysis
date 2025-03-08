import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

from transformers import pipeline
import matplotlib.pyplot as plt
import json

# Load the pre-trained emotion detection pipeline
@st.cache_resource
def load_emotion_analyzer():
    try:
        return pipeline("text-classification", 
                       model="j-hartmann/emotion-english-distilroberta-base", 
                       return_all_scores=True)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize the emotion analyze    
emotion_analyzer = load_emotion_analyzer()

def detect_emotion(text):
    """Analyze emotions in a single text input."""
    try:
        results = emotion_analyzer(text)
        emotions = {item['label']: item['score'] for item in results[0]}
        dominant_emotion = max(emotions, key=emotions.get)
        return dominant_emotion, emotions
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return None, None

def analyze_batch(texts):
    """Analyze emotions in multiple texts."""
    batch_results = []
    for text in texts:
        if text.strip():  # Skip empty texts
            emotion, scores = detect_emotion(text.strip())
            if emotion and scores:
                batch_results.append({
                    "text": text.strip(),
                    "dominant_emotion": emotion,
                    "scores": scores
                })
    return batch_results

def save_results_to_file(results, filename="emotion_results.json"):
    """Save analysis results to a JSON file."""
    try:
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(results, file, indent=4)
        st.success(f"Results saved to {filename}")
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")

def plot_emotion_scores(scores):
    """Create a bar chart of emotion scores."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(scores.keys(), scores.values(), color="skyblue")
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Scores")
        ax.set_title("Emotion Scores")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
            
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def main():
    st.title("📊 Emotion Detection with Streamlit")
    st.write("Analyze the emotions conveyed in text using a pre-trained model.")
    
    if not emotion_analyzer:
        st.error("Failed to load the emotion analyzer. Please check your installation and try again.")
        return
    
    # Sidebar navigation
    option = st.sidebar.radio(
        "Choose an option:",
        ("Analyze Single Text", "Analyze Multiple Texts", "About")
    )
    
    if option == "Analyze Single Text":
        st.header("Single Text Analysis")
        text = st.text_area("Enter text to analyze emotion:", height=150)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("Analyze")
            
        if analyze_button:
            if text.strip():
                with st.spinner("Analyzing text..."):
                    emotion, scores = detect_emotion(text)
                    if emotion and scores:
                        st.success(f"Detected Emotion: {emotion.upper()}")
                        
                        # Display scores and plot side by side
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.write("Emotion Scores:")
                            st.json(scores)
                        with col2:
                            fig = plot_emotion_scores(scores)
                            if fig:
                                st.pyplot(fig)
            else:
                st.warning("Please enter some text to analyze.")
    
    elif option == "Analyze Multiple Texts":
        st.header("Batch Text Analysis")
        st.info("Enter multiple texts separated by semicolons (;)")
        text_input = st.text_area("Enter texts:", height=150)
        
        if st.button("Analyze Batch"):
            if text_input.strip():
                with st.spinner("Analyzing texts..."):
                    texts = [t.strip() for t in text_input.split(";") if t.strip()]
                    if texts:
                        results = analyze_batch(texts)
                        if results:
                            for result in results:
                                with st.expander(f"Analysis for: {result['text'][:50]}..."):
                                    st.write(f"*Text:* {result['text']}")
                                    st.write(f"*Dominant Emotion:* {result['dominant_emotion'].upper()}")
                                    
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.write("*Emotion Scores:*")
                                        st.json(result['scores'])
                                    with col2:
                                        fig = plot_emotion_scores(result['scores'])
                                        if fig:
                                            st.pyplot(fig)
                            
                            if st.button("Save Results"):
                                save_results_to_file(results)
                    else:
                        st.warning("No valid texts found. Please check your input.")
            else:
                st.warning("Please enter some texts to analyze.")
    
    elif option == "About":
        st.header("About the App")
        st.write("""
        ## Emotion Detection App
        
        This application uses a pre-trained emotion detection model to analyze text and determine the emotional content. 
        
        ### Features:
        - Single text analysis
        - Batch text analysis
        - Visualization of emotion scores
        - Export results to JSON
        
        ### Model Information:
        This app uses the emotion-english-distilroberta-base model by J. Hartmann, which can detect various emotions in English text.
        
        ### How to Use:
        1. Choose your analysis mode from the sidebar
        2. Enter your text(s)
        3. Click analyze to see the results
        4. For batch analysis, you can save the results to a file
        
        ### Note:
        The model works best with English text and may not be accurate for other languages or very short texts.
        """)

if __name__ == "__main__":
    main()
