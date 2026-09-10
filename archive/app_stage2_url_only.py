import streamlit as st
import pickle
from scipy.sparse import hstack, csr_matrix
from features import tokenize_url, extract_lexical_features, check_website_html

with open('vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Phishing Website Detector")
st.write("Enter a URL below to analyze it for phishing indicators.")

url_input = st.text_input("URL:")

if st.button("Analyze"):
    if url_input:
        with st.spinner("Analyzing URL..."):
            tokens = tokenize_url(url_input)
            text_joint = ' '.join(tokens)
            X_vec = cv.transform([text_joint])
            
            lexical_features = extract_lexical_features(url_input)
            lexical_values = [
                lexical_features['has_ip_address'],
                lexical_features['slash_count'],
                lexical_features['has_port_number'],
                lexical_features['has_at_symbol'],
                lexical_features['hyphen_count']
            ]
            X_lex_sparse = csr_matrix([lexical_values], dtype=float)
            
            X_combined = hstack([X_vec, X_lex_sparse])
            text_prediction = model.predict(X_combined)[0]
            
            html_features = check_website_html(url_input)
            
            red_flags = sum([
                lexical_features['has_ip_address'],
                lexical_features['has_port_number'],
                lexical_features['has_at_symbol'],
                lexical_features['hyphen_count'] >= 3,
                lexical_features['slash_count'] >= 6,
            ])
            
            if html_features is None:
                red_flags += 1
            else:
                red_flags += sum([
                    html_features['external_link_ratio'] > 0.5,
                    html_features['has_hidden_iframe'],
                    html_features['has_suspicious_form'],
                    html_features['blocks_right_click'],
                ])
                
            if text_prediction == 'bad' and red_flags >= 1:
                verdict = 'bad'
            elif red_flags >= 2:
                verdict = 'bad'
            else:
                verdict = 'good'
                
            st.subheader("Analysis Results")
            if verdict == 'bad':
                st.error("Verdict: BAD (Potential Phishing Detected)")
            else:
                st.success("Verdict: GOOD (Looks Safe)")
                
            st.write(f"**Base Text Prediction:** {text_prediction}")
            st.write(f"**Total Red Flags Detected:** {red_flags}")
            
            st.json({"Lexical Features": lexical_features, "HTML Features": html_features})
    else:
        st.warning("Please enter a URL first.")