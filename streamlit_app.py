import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import csv
import os
from urllib.parse import urljoin, urlparse
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Configure Streamlit page
st.set_page_config(
    page_title="SaasQuatch Leads Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

class LeadScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_website_contacts(self, url, max_pages=5):
        """Extract contact information from a website"""
        contacts = []
        try:
            # Normalize URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, response.text, re.IGNORECASE)
            
            # Extract phone numbers
            phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            phones = re.findall(phone_pattern, response.text)
            
            # Extract company info
            title = soup.find('title')
            company_name = title.text.strip() if title else urlparse(url).netloc
            
            # Extract social media links
            social_links = {}
            for link in soup.find_all('a', href=True):
                href = link['href'].lower()
                if 'linkedin.com' in href:
                    social_links['linkedin'] = link['href']
                elif 'twitter.com' in href or 'x.com' in href:
                    social_links['twitter'] = link['href']
                elif 'facebook.com' in href:
                    social_links['facebook'] = link['href']
            
            # Create contact records
            for email in set(emails):
                contact = {
                    'source_url': url,
                    'company_name': company_name,
                    'email': email,
                    'phone': phones[0] if phones else '',
                    'linkedin': social_links.get('linkedin', ''),
                    'twitter': social_links.get('twitter', ''),
                    'facebook': social_links.get('facebook', ''),
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'lead_score': np.random.randint(1, 101)  # Placeholder score
                }
                contacts.append(contact)
            
            # If no emails found, still create a company record
            if not emails:
                contact = {
                    'source_url': url,
                    'company_name': company_name,
                    'email': '',
                    'phone': phones[0] if phones else '',
                    'linkedin': social_links.get('linkedin', ''),
                    'twitter': social_links.get('twitter', ''),
                    'facebook': social_links.get('facebook', ''),
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'lead_score': np.random.randint(1, 101)
                }
                contacts.append(contact)
                
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
        
        return contacts
    
    def search_google_leads(self, query, num_results=20):
        """Simulate Google search for leads (replace with actual API if available)"""
        # This is a placeholder - in production, you'd use Google Search API
        sample_domains = [
            'example-company.com', 'tech-startup.io', 'business-solutions.net',
            'digital-agency.com', 'consulting-firm.org', 'software-dev.co',
            'marketing-agency.net', 'ecommerce-store.com', 'fintech-startup.io',
            'healthcare-tech.com'
        ]
        
        leads = []
        for i, domain in enumerate(sample_domains[:num_results]):
            lead = {
                'source': 'Google Search',
                'query': query,
                'domain': domain,
                'company_name': domain.replace('-', ' ').replace('.com', '').replace('.io', '').replace('.net', '').replace('.org', '').replace('.co', '').title(),
                'estimated_employees': np.random.choice(['1-10', '11-50', '51-200', '201-500', '500+']),
                'industry': np.random.choice(['Technology', 'Marketing', 'Consulting', 'E-commerce', 'Healthcare', 'Finance']),
                'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'lead_score': np.random.randint(1, 101)
            }
            leads.append(lead)
        
        return leads

class LeadMLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        features = df.copy()
        
        # Create numerical features
        features['has_email'] = (features.get('email', '') != '').astype(int)
        features['has_phone'] = (features.get('phone', '') != '').astype(int)
        features['has_linkedin'] = (features.get('linkedin', '') != '').astype(int)
        features['has_social_media'] = ((features.get('twitter', '') != '') | 
                                      (features.get('facebook', '') != '')).astype(int)
        features['domain_length'] = features.get('source_url', '').str.len()
        features['company_name_length'] = features.get('company_name', '').str.len()
        
        # Encode categorical features
        categorical_features = ['industry', 'estimated_employees']
        for feature in categorical_features:
            if feature in features.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    features[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(features[feature].fillna('Unknown'))
                else:
                    features[f'{feature}_encoded'] = self.label_encoders[feature].transform(features[feature].fillna('Unknown'))
        
        # Select final features
        feature_columns = [
            'has_email', 'has_phone', 'has_linkedin', 'has_social_media',
            'domain_length', 'company_name_length', 'lead_score'
        ]
        
        # Add encoded categorical features
        for feature in categorical_features:
            if f'{feature}_encoded' in features.columns:
                feature_columns.append(f'{feature}_encoded')
        
        return features[feature_columns].fillna(0)
    
    def train_model(self, df):
        """Train the ML model"""
        features = self.prepare_features(df)
        
        # Create target variable (high quality lead based on lead_score)
        target = (df.get('lead_score', 50) > 70).astype(int)
        
        if len(features) < 2:
            raise ValueError("Need at least 2 samples to train the model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target if len(np.unique(target)) > 1 else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': dict(zip(features.columns, self.model.feature_importances_))
        }
    
    def predict_lead_quality(self, df):
        """Predict lead quality for new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict_proba(features_scaled)[:, 1]  # Probability of high quality
        
        return predictions

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 SaasQuatch Leads Enhanced</h1>
        <p>Advanced Lead Generation with ML-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = []
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = LeadMLModel()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🔍 Data Scraping",
        "🤖 ML Analysis",
        "📊 Dashboard",
        "📁 Data Management"
    ])
    
    if page == "🔍 Data Scraping":
        scraping_page()
    elif page == "🤖 ML Analysis":
        ml_analysis_page()
    elif page == "📊 Dashboard":
        dashboard_page()
    elif page == "📁 Data Management":
        data_management_page()

def scraping_page():
    st.header("🔍 Lead Data Scraping")
    
    scraper = LeadScraper()
    
    # Scraping options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Website Scraping")
        url_input = st.text_input("Enter website URL:", placeholder="https://example.com")
        
        if st.button("Scrape Website"):
            if url_input:
                with st.spinner("Scraping website..."):
                    contacts = scraper.scrape_website_contacts(url_input)
                    st.session_state.scraped_data.extend(contacts)
                    st.success(f"Scraped {len(contacts)} contacts from {url_input}")
                    
                    if contacts:
                        st.dataframe(pd.DataFrame(contacts))
    
    with col2:
        st.subheader("Google Search Leads")
        query_input = st.text_input("Enter search query:", placeholder="technology companies")
        num_results = st.slider("Number of results:", 5, 50, 20)
        
        if st.button("Search Google"):
            if query_input:
                with st.spinner("Searching for leads..."):
                    leads = scraper.search_google_leads(query_input, num_results)
                    st.session_state.scraped_data.extend(leads)
                    st.success(f"Found {len(leads)} potential leads")
                    
                    if leads:
                        st.dataframe(pd.DataFrame(leads))
    
    # Bulk URL scraping
    st.subheader("Bulk URL Scraping")
    urls_text = st.text_area("Enter URLs (one per line):", height=100)
    
    if st.button("Scrape All URLs"):
        if urls_text:
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            progress_bar = st.progress(0)
            
            all_contacts = []
            for i, url in enumerate(urls):
                with st.spinner(f"Scraping {url}..."):
                    contacts = scraper.scrape_website_contacts(url)
                    all_contacts.extend(contacts)
                    progress_bar.progress((i + 1) / len(urls))
                    time.sleep(1)  # Rate limiting
            
            st.session_state.scraped_data.extend(all_contacts)
            st.success(f"Scraped {len(all_contacts)} contacts from {len(urls)} websites")
    
    # Show current data
    if st.session_state.scraped_data:
        st.subheader("Current Scraped Data")
        df = pd.DataFrame(st.session_state.scraped_data)
        st.dataframe(df)
        
        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"scraped_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def ml_analysis_page():
    st.header("🤖 Machine Learning Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    data_source = st.radio("Choose data source:", ["Upload CSV", "Use Scraped Data"])
    
    df = None
    if data_source == "Upload CSV" and uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif data_source == "Use Scraped Data" and st.session_state.scraped_data:
        df = pd.DataFrame(st.session_state.scraped_data)
    
    if df is not None and not df.empty:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Train ML Model")
            if st.button("Train Model"):
                try:
                    with st.spinner("Training ML model..."):
                        results = st.session_state.ml_model.train_model(df)
                        
                        st.success("Model trained successfully!")
                        st.write(f"**Accuracy:** {results['accuracy']:.2%}")
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame(list(results['feature_importance'].items()), 
                                                   columns=['Feature', 'Importance'])
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                   title="Feature Importance in Lead Quality Prediction")
                        st.plotly_chart(fig)
                        
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
        
        with col2:
            st.subheader("Predict Lead Quality")
            if st.session_state.ml_model.is_trained:
                if st.button("Generate Predictions"):
                    try:
                        with st.spinner("Generating predictions..."):
                            predictions = st.session_state.ml_model.predict_lead_quality(df)
                            
                            df_with_predictions = df.copy()
                            df_with_predictions['ml_quality_score'] = predictions
                            df_with_predictions['quality_category'] = pd.cut(predictions, 
                                                                           bins=[0, 0.3, 0.7, 1.0], 
                                                                           labels=['Low', 'Medium', 'High'])
                            
                            st.success("Predictions generated!")
                            st.dataframe(df_with_predictions[['company_name', 'email', 'ml_quality_score', 'quality_category']])
                            
                            # Download predictions
                            csv = df_with_predictions.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions CSV",
                                data=csv,
                                file_name=f"lead_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
            else:
                st.warning("Please train the model first")
    else:
        st.warning("Please upload a CSV file or scrape some data first")

def dashboard_page():
    st.header("📊 Lead Analytics Dashboard")
    
    if st.session_state.scraped_data:
        df = pd.DataFrame(st.session_state.scraped_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Leads", len(df))
        with col2:
            st.metric("Companies with Email", len(df[df.get('email', '') != '']))
        with col3:
            st.metric("Companies with Phone", len(df[df.get('phone', '') != '']))
        with col4:
            avg_score = df.get('lead_score', [0]).mean() if 'lead_score' in df.columns else 0
            st.metric("Avg Lead Score", f"{avg_score:.1f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'lead_score' in df.columns:
                fig = px.histogram(df, x='lead_score', title='Lead Score Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'industry' in df.columns:
                industry_counts = df['industry'].value_counts()
                fig = px.pie(values=industry_counts.values, names=industry_counts.index, 
                           title='Leads by Industry')
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Detailed Lead Information")
        st.dataframe(df)
        
    else:
        st.warning("No data available. Please scrape some leads first.")

def data_management_page():
    st.header("📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Data")
        if st.session_state.scraped_data:
            df = pd.DataFrame(st.session_state.scraped_data)
            st.write(f"Total records: {len(df)}")
            st.dataframe(df.head())
            
            if st.button("Clear All Data"):
                st.session_state.scraped_data = []
                st.success("All data cleared!")
                st.rerun()
        else:
            st.info("No data available")
    
    with col2:
        st.subheader("Export Options")
        if st.session_state.scraped_data:
            df = pd.DataFrame(st.session_state.scraped_data)
            
            # CSV Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # JSON Export
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_data,
                file_name=f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()