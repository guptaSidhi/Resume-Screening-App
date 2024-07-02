import streamlit as st
import pickle
import re
import nltk
import string

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

nltk.download('punkt')
nltk.download('stopwords')

ab = pickle.load(open('ab.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def clean_text(txt):
    
    cleantxt = re.sub(r'https?://\S+|www\.\S+', ' ', txt)
    cleantxt = re.sub(r'@\S+', ' ', cleantxt)
    cleantxt = re.sub(r'#\S+', ' ', cleantxt)
    cleantxt = re.sub('[\r\n]+',' ',cleantxt)
    
    return cleantxt

symbols = string.punctuation

def remove_punctuation(txt):
    for char in symbols:
        txt = txt.replace(char,'')
    return txt

def remove_stopwords(txt):
    cleantxt = []
    for word in txt.split():
        if word in stop_words:
            cleantxt.append('')
        else:
            cleantxt.append(word)
            
    return " ".join(cleantxt)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stemming_txt(txt):
    return ps.stem(txt)

category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_text(resume_text)
        cleaned_resume = remove_punctuation(cleaned_resume)
        cleaned_resume = remove_stopwords(cleaned_resume)
        cleaned_resume = stemming_txt(cleaned_resume)

        input_features = tfidf.transform([cleaned_resume])

        prediction_id = ab.predict(input_features)[0]

        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()