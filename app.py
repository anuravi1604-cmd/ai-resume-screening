import streamlit as st
from preprocess import extract_text_from_pdf, clean_text, extract_skills
from model import predict_role, load_models

# Configure Streamlit page
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="centered"
)

# Load model.pkl and vectorizer.pkl using caching for fast interactive usage
@st.cache_resource
def init_models():
    return load_models()

def main():
    st.title("📄 AI Resume Screener")
    st.write("Upload a PDF resume to automatically predict the best-fit job role and extract listed skills.")
    
    # Trigger model loading
    init_models()
    
    # Accept resume upload
    uploaded_file = st.file_uploader("Upload Resume (PDF format only)", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            # Extract text
            raw_text = extract_text_from_pdf(uploaded_file)
            
        if not raw_text:
            st.error("Could not extract text from the uploaded PDF. Ensure the PDF is not an image.")
            st.stop()
            
        with st.spinner("Analyzing resume content..."):
            # Preprocess text
            cleaned_text = clean_text(raw_text)
            
            # Predict job role using predict_role()
            # (predict_role internally handles its own cleaning as well, handling edge cases)
            predicted_role, confidence = predict_role(cleaned_text)
            
            # Extract skills based on raw text for exact capitalization match
            found_skills = extract_skills(raw_text)
            
        # Display predicted category clearly
        st.success("Analysis Complete!")
        
        st.subheader("🤖 Prediction Results")
        # Clearly display the category as requested
        st.markdown(f"### **Predicted Category: {predicted_role}**")
        st.metric("Confidence Score", f"{confidence * 100:.2f}%")
            
        st.subheader("💡 Extracted Skills")
        if found_skills:
            st.write(", ".join(found_skills))
        else:
            st.write("No pre-defined skills found in the text.")
            
        with st.expander("Show Extracted Resume Text"):
            st.text_area("Extracted Text", raw_text, height=250)

if __name__ == "__main__":
    main()
