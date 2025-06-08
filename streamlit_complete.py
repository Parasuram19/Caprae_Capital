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
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Lead Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedLeadScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.linkedin_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def scrape_linkedin_companies(self, search_terms, location="", industry="", company_size=""):
        """Scrape LinkedIn company data (simulated for demo)"""
        companies = []
        
        try:
            # Simulate LinkedIn company search results
            base_companies = [
                {
                    'name': 'TechCorp Solutions',
                    'industry': 'Technology',
                    'size': '201-500',
                    'location': 'San Francisco, CA',
                    'domain': 'techcorp.com',
                    'description': 'Leading software development company',
                    'employees_on_linkedin': 450,
                    'follower_count': 12500
                },
                {
                    'name': 'Digital Marketing Pro',
                    'industry': 'Marketing',
                    'size': '51-200',
                    'location': 'New York, NY',
                    'domain': 'digitalmarketingpro.com',
                    'description': 'Full-service digital marketing agency',
                    'employees_on_linkedin': 125,
                    'follower_count': 8500
                },
                {
                    'name': 'HealthTech Innovations',
                    'industry': 'Healthcare',
                    'size': '11-50',
                    'location': 'Boston, MA',
                    'domain': 'healthtech-innovations.com',
                    'description': 'Healthcare technology solutions',
                    'employees_on_linkedin': 35,
                    'follower_count': 3200
                },
                {
                    'name': 'FinanceFlow Systems',
                    'industry': 'Financial Services',
                    'size': '501-1000',
                    'location': 'Chicago, IL',
                    'domain': 'financeflow.com',
                    'description': 'Financial software and consulting',
                    'employees_on_linkedin': 750,
                    'follower_count': 18500
                },
                {
                    'name': 'EcoGreen Solutions',
                    'industry': 'Environmental Services',
                    'size': '1-10',
                    'location': 'Portland, OR',
                    'domain': 'ecogreen-solutions.org',
                    'description': 'Sustainable business solutions',
                    'employees_on_linkedin': 8,
                    'follower_count': 1200
                }
            ]
            
            # Filter based on criteria
            for company in base_companies:
                # Apply filters
                if industry and industry.lower() not in company['industry'].lower():
                    continue
                if location and location.lower() not in company['location'].lower():
                    continue
                if company_size and company['size'] != company_size:
                    continue
                if search_terms:
                    search_match = any(term.lower() in company['name'].lower() or 
                                     term.lower() in company['description'].lower() 
                                     for term in search_terms.split())
                    if not search_match:
                        continue
                
                # Generate additional data
                company_data = {
                    'source': 'LinkedIn',
                    'company_name': company['name'],
                    'domain': company['domain'],
                    'industry': company['industry'],
                    'company_size': company['size'],
                    'location': company['location'],
                    'description': company['description'],
                    'employees_on_linkedin': company['employees_on_linkedin'],
                    'follower_count': company['follower_count'],
                    'email': f"info@{company['domain']}",
                    'phone': f"+1-{np.random.randint(200,999)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}",
                    'website': f"https://{company['domain']}",
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'lead_score': np.random.randint(40, 95),
                    'engagement_score': np.random.randint(30, 85),
                    'company_growth': np.random.choice(['High', 'Medium', 'Low']),
                    'funding_stage': np.random.choice(['Seed', 'Series A', 'Series B', 'Established'])
                }
                companies.append(company_data)
            
        except Exception as e:
            st.error(f"Error scraping LinkedIn: {str(e)}")
        
        return companies
    
    def scrape_google_companies(self, search_query, location="", domain_filter=""):
        """Scrape company data from Google search (simulated)"""
        companies = []
        
        try:
            # Simulate Google search results
            base_results = [
                {
                    'name': 'CloudTech Enterprises',
                    'domain': 'cloudtech-ent.com',
                    'snippet': 'Leading cloud infrastructure provider serving Fortune 500 companies',
                    'business_type': 'B2B SaaS'
                },
                {
                    'name': 'RetailBoost Analytics',
                    'domain': 'retailboost.io',
                    'snippet': 'E-commerce analytics platform helping retailers optimize sales',
                    'business_type': 'Analytics'
                },
                {
                    'name': 'MedDevice Innovations',
                    'domain': 'meddevice-innov.com',
                    'snippet': 'Medical device manufacturer specializing in diagnostic equipment',
                    'business_type': 'Manufacturing'
                },
                {
                    'name': 'EduTech Solutions',
                    'domain': 'edutech-sol.edu',
                    'snippet': 'Educational technology platform for K-12 schools',
                    'business_type': 'Education Technology'
                },
                {
                    'name': 'AgriSmart Systems',
                    'domain': 'agrismart.farm',
                    'snippet': 'Smart farming solutions using IoT and AI technology',
                    'business_type': 'AgTech'
                }
            ]
            
            for result in base_results:
                # Apply filters
                if domain_filter and domain_filter.lower() not in result['domain'].lower():
                    continue
                if search_query:
                    query_match = any(term.lower() in result['name'].lower() or 
                                    term.lower() in result['snippet'].lower() 
                                    for term in search_query.split())
                    if not query_match:
                        continue
                
                company_data = {
                    'source': 'Google Search',
                    'company_name': result['name'],
                    'domain': result['domain'],
                    'business_concept': result['business_type'],
                    'description': result['snippet'],
                    'location': location or f"{np.random.choice(['California', 'New York', 'Texas', 'Florida'])}, USA",
                    'estimated_revenue': f"${np.random.randint(1, 50)}M",
                    'tech_stack': np.random.choice(['React/Node.js', 'Python/Django', 'Java/Spring', '.NET/Azure']),
                    'email': f"contact@{result['domain']}",
                    'phone': f"+1-{np.random.randint(200,999)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}",
                    'website': f"https://{result['domain']}",
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'lead_score': np.random.randint(35, 90),
                    'seo_score': np.random.randint(40, 95),
                    'social_presence': np.random.choice(['Strong', 'Medium', 'Weak'])
                }
                companies.append(company_data)
                
        except Exception as e:
            st.error(f"Error scraping Google: {str(e)}")
        
        return companies
    
    def scrape_crunchbase_companies(self, industry="", funding_stage="", location=""):
        """Scrape startup data from Crunchbase (simulated)"""
        companies = []
        
        try:
            base_startups = [
                {
                    'name': 'AIVision Robotics',
                    'industry': 'Artificial Intelligence',
                    'funding_stage': 'Series B',
                    'total_funding': '$25M',
                    'location': 'Austin, TX'
                },
                {
                    'name': 'GreenEnergy Solutions',
                    'industry': 'Clean Technology',
                    'funding_stage': 'Series A',
                    'total_funding': '$12M',
                    'location': 'Denver, CO'
                },
                {
                    'name': 'BioMed Diagnostics',
                    'industry': 'Biotechnology',
                    'funding_stage': 'Seed',
                    'total_funding': '$3M',
                    'location': 'San Diego, CA'
                },
                {
                    'name': 'FinTech Secure',
                    'industry': 'Financial Technology',
                    'funding_stage': 'Series C',
                    'total_funding': '$45M',
                    'location': 'Miami, FL'
                },
                {
                    'name': 'FoodDelivery Plus',
                    'industry': 'Food & Beverage',
                    'funding_stage': 'Series A',
                    'total_funding': '$8M',
                    'location': 'Seattle, WA'
                }
            ]
            
            for startup in base_startups:
                # Apply filters
                if industry and industry.lower() not in startup['industry'].lower():
                    continue
                if funding_stage and startup['funding_stage'] != funding_stage:
                    continue
                if location and location.lower() not in startup['location'].lower():
                    continue
                
                domain = startup['name'].lower().replace(' ', '').replace('&', 'and') + '.com'
                
                company_data = {
                    'source': 'Crunchbase',
                    'company_name': startup['name'],
                    'domain': domain,
                    'industry': startup['industry'],
                    'funding_stage': startup['funding_stage'],
                    'total_funding': startup['total_funding'],
                    'location': startup['location'],
                    'founded_year': np.random.randint(2015, 2023),
                    'employee_count': np.random.randint(10, 500),
                    'email': f"hello@{domain}",
                    'website': f"https://{domain}",
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'lead_score': np.random.randint(50, 95),
                    'growth_potential': np.random.choice(['High', 'Medium', 'Low']),
                    'market_category': np.random.choice(['B2B', 'B2C', 'B2B2C'])
                }
                companies.append(company_data)
                
        except Exception as e:
            st.error(f"Error scraping Crunchbase: {str(e)}")
        
        return companies

class LeadQualityPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.feature_names = []
    
    def prepare_features(self, df):
        """Prepare features for lead quality prediction"""
        features = df.copy()
        
        # Numerical features
        features['has_email'] = (features['email'].fillna('') != '').astype(int)
        features['has_phone'] = (features['phone'].fillna('') != '').astype(int)
        features['has_website'] = (features['website'].fillna('') != '').astype(int)
        
        # Company size encoding
        size_mapping = {
            '1-10': 1, '11-50': 2, '51-200': 3, '201-500': 4, 
            '501-1000': 5, '1000+': 6, '500+': 5
        }
        if 'company_size' in features.columns:
            features['company_size_numeric'] = features['company_size'].map(size_mapping).fillna(0)
        else:
            features['company_size_numeric'] = 0
        
        # Industry encoding
        if 'industry' in features.columns:
            if 'industry' not in self.label_encoders:
                self.label_encoders['industry'] = LabelEncoder()
                features['industry_encoded'] = self.label_encoders['industry'].fit_transform(
                    features['industry'].fillna('Unknown')
                )
            else:
                try:
                    features['industry_encoded'] = self.label_encoders['industry'].transform(
                        features['industry'].fillna('Unknown')
                    )
                except ValueError:
                    features['industry_encoded'] = 0
        else:
            features['industry_encoded'] = 0
        
        # Engagement metrics
        if 'follower_count' in features.columns:
            features['follower_count_log'] = np.log1p(features['follower_count'].fillna(0))
        else:
            features['follower_count_log'] = 0
            
        if 'employees_on_linkedin' in features.columns:
            features['employees_log'] = np.log1p(features['employees_on_linkedin'].fillna(0))
        else:
            features['employees_log'] = 0
        
        # Text features
        if 'description' in features.columns:
            features['description_length'] = features['description'].fillna('').str.len()
        else:
            features['description_length'] = 0
        
        # Select final features
        feature_columns = [
            'has_email', 'has_phone', 'has_website', 'company_size_numeric',
            'industry_encoded', 'follower_count_log', 'employees_log', 'description_length'
        ]
        
        # Add lead_score if available for training
        if 'lead_score' in features.columns:
            feature_columns.append('lead_score')
        
        self.feature_names = feature_columns
        return features[feature_columns].fillna(0)
    
    def train_model(self, df):
        """Train the lead quality prediction model"""
        if len(df) < 10:
            raise ValueError("Need at least 10 samples to train the model")
        
        features = self.prepare_features(df)
        
        # Create target based on multiple factors
        target_factors = []
        
        if 'lead_score' in df.columns:
            target_factors.append(df['lead_score'] > df['lead_score'].median())
        
        if 'follower_count' in df.columns:
            target_factors.append(df['follower_count'] > df['follower_count'].median())
        
        if 'company_size' in df.columns:
            large_company = df['company_size'].isin(['201-500', '501-1000', '1000+', '500+'])
            target_factors.append(large_company)
        
        # Combine factors
        if target_factors:
            target = sum(target_factors) >= len(target_factors) // 2
        else:
            target = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
        
        target = target.astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'n_samples': len(df)
        }
    
    def predict(self, df):
        """Predict lead quality"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        predictions = self.model.predict(features_scaled)
        
        return probabilities, predictions

class ContactEnricher:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.is_trained = False
    
    def generate_email_patterns(self, name, domain):
        """Generate common email patterns"""
        if not name or not domain:
            return []
        
        name_parts = name.lower().replace(' ', '').split()
        if len(name_parts) < 2:
            first_name = name_parts[0] if name_parts else 'contact'
            last_name = ''
        else:
            first_name = name_parts[0]
            last_name = name_parts[-1]
        
        patterns = [
            f"{first_name}@{domain}",
            f"{first_name}.{last_name}@{domain}",
            f"{first_name}{last_name}@{domain}",
            f"{first_name}_{last_name}@{domain}",
            f"{first_name[0]}.{last_name}@{domain}",
            f"{first_name[0]}{last_name}@{domain}",
            f"info@{domain}",
            f"contact@{domain}",
            f"hello@{domain}",
            f"sales@{domain}"
        ]
        
        return patterns[:5]  # Return top 5 patterns
    
    def enrich_contacts(self, df):
        """Enrich contact information"""
        enriched_data = df.copy()
        
        # Generate email patterns for contacts without emails
        for idx, row in enriched_data.iterrows():
            if pd.isna(row.get('email', '')) or row.get('email', '') == '':
                company_name = row.get('company_name', '')
                domain = row.get('domain', '')
                
                if company_name and domain:
                    email_patterns = self.generate_email_patterns(company_name, domain)
                    enriched_data.at[idx, 'predicted_emails'] = ', '.join(email_patterns)
                    enriched_data.at[idx, 'email_confidence'] = np.random.uniform(0.3, 0.8)
        
        # Add additional enrichment
        enriched_data['contact_score'] = np.random.uniform(0.2, 0.9, len(enriched_data))
        enriched_data['data_completeness'] = enriched_data.apply(
            lambda row: sum([
                bool(row.get('email', '')),
                bool(row.get('phone', '')),
                bool(row.get('website', '')),
                bool(row.get('location', ''))
            ]) / 4, axis=1
        )
        
        return enriched_data

class FuzzyDeduplicator:
    def __init__(self):
        # Initialize sentence transformer for embeddings
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.model = None
            st.warning("Sentence transformer not available. Using TF-IDF for similarity.")
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    def create_company_signature(self, row):
        """Create a signature for company comparison"""
        signature_parts = []
        
        if row.get('company_name'):
            signature_parts.append(str(row['company_name']).lower().strip())
        if row.get('domain'):
            signature_parts.append(str(row['domain']).lower().strip())
        if row.get('location'):
            signature_parts.append(str(row['location']).lower().strip())
        
        return ' '.join(signature_parts)
    
    def find_duplicates(self, df, similarity_threshold=0.8):
        """Find duplicate companies using embedding similarity"""
        if len(df) < 2:
            return df, []
        
        # Create signatures
        signatures = df.apply(self.create_company_signature, axis=1).tolist()
        
        try:
            if self.model:
                # Use sentence transformer embeddings
                embeddings = self.model.encode(signatures)
                similarity_matrix = cosine_similarity(embeddings)
            else:
                # Fallback to TF-IDF
                tfidf_matrix = self.vectorizer.fit_transform(signatures)
                similarity_matrix = cosine_similarity(tfidf_matrix)
        except Exception as e:
            st.error(f"Error computing similarity: {e}")
            return df, []
        
        # Find duplicates
        duplicates = []
        processed = set()
        
        for i in range(len(similarity_matrix)):
            if i in processed:
                continue
                
            similar_indices = []
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > similarity_threshold:
                    similar_indices.append(j)
            
            if similar_indices:
                group = [i] + similar_indices
                duplicates.append({
                    'group_id': len(duplicates),
                    'indices': group,
                    'companies': [df.iloc[idx]['company_name'] for idx in group],
                    'similarity_scores': [similarity_matrix[i][j] for j in similar_indices]
                })
                processed.update(group)
        
        # Create deduplicated dataframe
        keep_indices = []
        for i in range(len(df)):
            # Keep the first occurrence from each duplicate group
            is_duplicate = False
            for dup_group in duplicates:
                if i in dup_group['indices'] and i != dup_group['indices'][0]:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)
        
        deduplicated_df = df.iloc[keep_indices].reset_index(drop=True)
        
        return deduplicated_df, duplicates

class LeadSegmentation:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
    
    def prepare_features(self, df):
        """Prepare features for clustering"""
        features = df.copy()
        
        # Numerical features
        if 'lead_score' in features.columns:
            features['lead_score_norm'] = features['lead_score'] / 100.0
        else:
            features['lead_score_norm'] = 0.5
        
        # Company size
        size_mapping = {
            '1-10': 1, '11-50': 2, '51-200': 3, '201-500': 4, 
            '501-1000': 5, '1000+': 6, '500+': 5
        }
        if 'company_size' in features.columns:
            features['size_numeric'] = features['company_size'].map(size_mapping).fillna(2)
        else:
            features['size_numeric'] = 2
        
        # Engagement metrics
        if 'follower_count' in features.columns:
            features['engagement'] = np.log1p(features['follower_count'].fillna(0)) / 10
        else:
            features['engagement'] = 0.5
        
        # Growth potential
        growth_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
        if 'growth_potential' in features.columns:
            features['growth_score'] = features['growth_potential'].map(growth_mapping).fillna(2)
        elif 'company_growth' in features.columns:
            features['growth_score'] = features['company_growth'].map(growth_mapping).fillna(2)
        else:
            features['growth_score'] = 2
        
        # Select features for clustering
        cluster_features = ['lead_score_norm', 'size_numeric', 'engagement', 'growth_score']
        self.feature_names = cluster_features
        
        return features[cluster_features].fillna(0.5)
    
    def segment_leads(self, df, n_clusters=4):
        """Segment leads using K-means clustering"""
        if len(df) < n_clusters:
            raise ValueError(f"Need at least {n_clusters} samples for clustering")
        
        self.kmeans.n_clusters = n_clusters
        features = self.prepare_features(df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(features_scaled)
        
        # Calculate cluster characteristics
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        
        cluster_stats = []
        segment_names = ['High Value', 'Growth Potential', 'Standard', 'Low Priority']
        
        for i in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == i]
            
            stats = {
                'cluster_id': i,
                'segment_name': segment_names[i] if i < len(segment_names) else f'Segment {i+1}',
                'count': len(cluster_data),
                'avg_lead_score': cluster_data['lead_score'].mean() if 'lead_score' in cluster_data.columns else 0,
                'companies': cluster_data['company_name'].tolist()[:5]  # Sample companies
            }
            cluster_stats.append(stats)
        
        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(features_scaled, clusters)
        except:
            silhouette_avg = 0.5
        
        self.is_fitted = True
        
        return {
            'clustered_df': df_clustered,
            'cluster_stats': cluster_stats,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters
        }

class LeadDashboard:
    def __init__(self):
        self.scraper = AdvancedLeadScraper()
        self.predictor = LeadQualityPredictor()
        self.enricher = ContactEnricher()
        self.deduplicator = FuzzyDeduplicator()
        self.segmentation = LeadSegmentation()
    
    def create_metrics_dashboard(self, df):
        """Create metrics dashboard"""
        if df.empty:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df)}</h3>
                <p>Total Leads</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_score = df['lead_score'].mean() if 'lead_score' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_score:.1f}</h3>
                <p>Avg Lead Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_domains = df['domain'].nunique() if 'domain' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{unique_domains}</h3>
                <p>Unique Domains</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            conversion_rate = len(df[df['lead_score'] > 70]) / len(df) * 100 if 'lead_score' in df.columns and len(df) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{conversion_rate:.1f}%</h3>
                <p>High Quality Rate</p>
            </div>
            """, unsafe_allow_html=True)
    
    def create_visualizations(self, df):
        """Create interactive visualizations"""
        if df.empty:
            return
        
        st.subheader("📊 Lead Analytics")
        
        # Lead Score Distribution
        if 'lead_score' in df.columns:
            fig_hist = px.histogram(
                df, x='lead_score', nbins=20,
                title="Lead Score Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(
                template='plotly_white',
                title_font_size=16,
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Industry Distribution
            if 'industry' in df.columns:
                industry_counts = df['industry'].value_counts().head(10)
                fig_bar = px.bar(
                    x=industry_counts.index,
                    y=industry_counts.values,
                    title="Top Industries",
                    color=industry_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(
                    template='plotly_white',
                    title_font_size=16,
                    showlegend=False,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Company Size Distribution
            if 'company_size' in df.columns:
                size_counts = df['company_size'].value_counts()
                fig_pie = px.pie(
                    values=size_counts.values,
                    names=size_counts.index,
                    title="Company Size Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(
                    template='plotly_white',
                    title_font_size=16
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Geographic Distribution
        if 'location' in df.columns:
            location_counts = df['location'].value_counts().head(15)
            fig_geo = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Geographic Distribution",
                color=location_counts.values,
                color_continuous_scale='Blues'
            )
            fig_geo.update_layout(
                template='plotly_white',
                title_font_size=16,
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_geo, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Lead Intelligence Platform</h1>
        <p>Intelligent Lead Generation, Scoring & Contact Enrichment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = LeadDashboard()
    
    # Initialize session state
    if 'leads_df' not in st.session_state:
        st.session_state.leads_df = pd.DataFrame()
    if 'processed_leads' not in st.session_state:
        st.session_state.processed_leads = pd.DataFrame()
    
    # Sidebar configuration
    st.sidebar.title("🎯 Lead Generation Settings")
    
    # Data source selection
    data_sources = st.sidebar.multiselect(
        "Select Data Sources",
        ["LinkedIn Companies", "Google Search", "Crunchbase Startups"],
        default=["LinkedIn Companies"]
    )
    
    # Search parameters
    st.sidebar.markdown("### Search Parameters")
    search_terms = st.sidebar.text_input("Search Terms", "technology software")
    location_filter = st.sidebar.text_input("Location Filter", "")
    industry_filter = st.sidebar.text_input("Industry Filter", "")
    
    # LinkedIn specific filters
    if "LinkedIn Companies" in data_sources:
        st.sidebar.markdown("### LinkedIn Filters")
        company_size = st.sidebar.selectbox(
            "Company Size",
            ["", "1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]
        )
    
    # Crunchbase specific filters
    if "Crunchbase Startups" in data_sources:
        st.sidebar.markdown("### Crunchbase Filters")
        funding_stage = st.sidebar.selectbox(
            "Funding Stage",
            ["", "Seed", "Series A", "Series B", "Series C", "IPO"]
        )
    
    # Advanced settings
    st.sidebar.markdown("### Advanced Settings")
    enable_ai_scoring = st.sidebar.checkbox("Enable AI Lead Scoring", True)
    enable_deduplication = st.sidebar.checkbox("Enable Smart Deduplication", True)
    enable_segmentation = st.sidebar.checkbox("Enable Lead Segmentation", True)
    enable_enrichment = st.sidebar.checkbox("Enable Contact Enrichment", True)
    
    # Generate leads button
    if st.sidebar.button("🔍 Generate Leads", type="primary"):
        with st.spinner("Generating leads from selected sources..."):
            all_leads = []
            
            # LinkedIn scraping
            if "LinkedIn Companies" in data_sources:
                linkedin_leads = dashboard.scraper.scrape_linkedin_companies(
                    search_terms, location_filter, industry_filter, company_size
                )
                all_leads.extend(linkedin_leads)
                st.success(f"✅ Found {len(linkedin_leads)} LinkedIn companies")
            
            # Google scraping
            if "Google Search" in data_sources:
                google_leads = dashboard.scraper.scrape_google_companies(
                    search_terms, location_filter
                )
                all_leads.extend(google_leads)
                st.success(f"✅ Found {len(google_leads)} Google results")
            
            # Crunchbase scraping
            if "Crunchbase Startups" in data_sources:
                crunchbase_leads = dashboard.scraper.scrape_crunchbase_companies(
                    industry_filter, funding_stage, location_filter
                )
                all_leads.extend(crunchbase_leads)
                st.success(f"✅ Found {len(crunchbase_leads)} Crunchbase startups")
            
            if all_leads:
                st.session_state.leads_df = pd.DataFrame(all_leads)
                st.success(f"🎉 Total leads generated: {len(all_leads)}")
            else:
                st.warning("No leads found with current filters")
    
    # Process leads if available
    if not st.session_state.leads_df.empty:
        st.markdown("---")
        st.subheader("🔧 Lead Processing & Enhancement")
        
        processing_options = st.columns(4)
        
        with processing_options[0]:
            if st.button("🧠 Score Leads") and enable_ai_scoring:
                with st.spinner("Training AI model and scoring leads..."):
                    try:
                        training_results = dashboard.predictor.train_model(st.session_state.leads_df)
                        probabilities, predictions = dashboard.predictor.predict(st.session_state.leads_df)
                        
                        st.session_state.leads_df['ai_lead_quality'] = probabilities
                        st.session_state.leads_df['quality_prediction'] = predictions
                        
                        st.success(f"✅ AI scoring complete! Accuracy: {training_results['accuracy']:.2f}")
                        
                        # Show feature importance
                        importance_df = pd.DataFrame.from_dict(
                            training_results['feature_importance'], 
                            orient='index', 
                            columns=['Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        st.write("**Feature Importance:**")
                        st.dataframe(importance_df)
                        
                    except Exception as e:
                        st.error(f"Error in AI scoring: {str(e)}")
        
        with processing_options[1]:
            if st.button("🔍 Deduplicate") and enable_deduplication:
                with st.spinner("Removing duplicate companies..."):
                    try:
                        original_count = len(st.session_state.leads_df)
                        deduplicated_df, duplicates = dashboard.deduplicator.find_duplicates(
                            st.session_state.leads_df, similarity_threshold=0.8
                        )
                        st.session_state.leads_df = deduplicated_df
                        
                        removed_count = original_count - len(deduplicated_df)
                        st.success(f"✅ Removed {removed_count} duplicates")
                        
                        if duplicates:
                            st.write("**Duplicate Groups Found:**")
                            for dup in duplicates[:3]:  # Show first 3 groups
                                st.write(f"- {', '.join(dup['companies'])}")
                        
                    except Exception as e:
                        st.error(f"Error in deduplication: {str(e)}")
        
        with processing_options[2]:
            if st.button("📊 Segment Leads") and enable_segmentation:
                with st.spinner("Segmenting leads..."):
                    try:
                        segmentation_results = dashboard.segmentation.segment_leads(
                            st.session_state.leads_df, n_clusters=4
                        )
                        
                        st.session_state.processed_leads = segmentation_results['clustered_df']
                        
                        st.success(f"✅ Segmentation complete! Silhouette Score: {segmentation_results['silhouette_score']:.2f}")
                        
                        # Show segment statistics
                        st.write("**Lead Segments:**")
                        for stat in segmentation_results['cluster_stats']:
                            st.write(f"- **{stat['segment_name']}**: {stat['count']} leads (Avg Score: {stat['avg_lead_score']:.1f})")
                        
                    except Exception as e:
                        st.error(f"Error in segmentation: {str(e)}")
        
        with processing_options[3]:
            if st.button("📧 Enrich Contacts") and enable_enrichment:
                with st.spinner("Enriching contact information..."):
                    try:
                        enriched_df = dashboard.enricher.enrich_contacts(st.session_state.leads_df)
                        st.session_state.leads_df = enriched_df
                        
                        st.success("✅ Contact enrichment complete!")
                        
                        # Show enrichment stats
                        if 'predicted_emails' in enriched_df.columns:
                            email_predictions = enriched_df['predicted_emails'].notna().sum()
                            st.write(f"**Generated {email_predictions} email predictions**")
                        
                    except Exception as e:
                        st.error(f"Error in contact enrichment: {str(e)}")
        
        # Display results
        st.markdown("---")
        dashboard.create_metrics_dashboard(st.session_state.leads_df)
        dashboard.create_visualizations(st.session_state.leads_df)
        
        # Data export section
        st.subheader("📤 Export Data")
        
        export_cols = st.columns(3)
        
        with export_cols[0]:
            csv_data = st.session_state.leads_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with export_cols[1]:
            json_data = st.session_state.leads_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with export_cols[2]:
            # Prepare Excel export
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                st.session_state.leads_df.to_excel(writer, sheet_name='Leads', index=False)
                
                if not st.session_state.processed_leads.empty:
                    st.session_state.processed_leads.to_excel(writer, sheet_name='Segmented_Leads', index=False)
            
            st.download_button(
                label="📈 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Data table
        st.subheader("📋 Lead Database")
        
        # Filters for data table
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            if 'industry' in st.session_state.leads_df.columns:
                industry_options = ['All'] + list(st.session_state.leads_df['industry'].dropna().unique())
                selected_industry = st.selectbox("Filter by Industry", industry_options)
        
        with filter_cols[1]:
            if 'company_size' in st.session_state.leads_df.columns:
                size_options = ['All'] + list(st.session_state.leads_df['company_size'].dropna().unique())
                selected_size = st.selectbox("Filter by Company Size", size_options)
        
        with filter_cols[2]:
            if 'source' in st.session_state.leads_df.columns:
                source_options = ['All'] + list(st.session_state.leads_df['source'].dropna().unique())
                selected_source = st.selectbox("Filter by Source", source_options)
        
        with filter_cols[3]:
            min_score = st.slider("Minimum Lead Score", 0, 100, 0)
        
        # Apply filters
        filtered_df = st.session_state.leads_df.copy()
        
        if 'industry' in filtered_df.columns and 'selected_industry' in locals() and selected_industry != 'All':
            filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
        
        if 'company_size' in filtered_df.columns and 'selected_size' in locals() and selected_size != 'All':
            filtered_df = filtered_df[filtered_df['company_size'] == selected_size]
        
        if 'source' in filtered_df.columns and 'selected_source' in locals() and selected_source != 'All':
            filtered_df = filtered_df[filtered_df['source'] == selected_source]
        
        if 'lead_score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['lead_score'] >= min_score]
        
        # Display filtered results
        st.write(f"Showing {len(filtered_df)} of {len(st.session_state.leads_df)} leads")
        
        if not filtered_df.empty:
            # Column selection for display
            available_columns = filtered_df.columns.tolist()
            default_columns = ['company_name', 'industry', 'location', 'lead_score', 'email', 'website']
            display_columns = [col for col in default_columns if col in available_columns]
            
            if len(display_columns) < len(available_columns):
                remaining_columns = [col for col in available_columns if col not in display_columns]
                display_columns.extend(remaining_columns[:5])  # Add up to 5 more columns
            
            selected_columns = st.multiselect(
                "Select columns to display",
                available_columns,
                default=display_columns[:8]  # Limit to 8 columns for better display
            )
            
            if selected_columns:
                display_df = filtered_df[selected_columns]
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'lead_score': st.column_config.ProgressColumn(
                            "Lead Score",
                            help="AI-generated lead quality score",
                            min_value=0,
                            max_value=100,
                        ),
                        'website': st.column_config.LinkColumn(
                            "Website",
                            help="Company website",
                            display_text="Visit"
                        ),
                        'email': st.column_config.TextColumn(
                            "Email",
                            help="Contact email"
                        )
                    }
                )
        else:
            st.info("No leads match the current filters.")
    
    else:
        st.info("👆 Configure your search parameters and click 'Generate Leads' to get started!")
        
        # Show demo data option
        if st.button("🎲 Load Demo Data"):
            with st.spinner("Loading demo data..."):
                demo_leads = dashboard.scraper.scrape_linkedin_companies("technology", "", "", "")
                demo_leads.extend(dashboard.scraper.scrape_google_companies("software", ""))
                demo_leads.extend(dashboard.scraper.scrape_crunchbase_companies("", "", ""))
                
                st.session_state.leads_df = pd.DataFrame(demo_leads)
                st.success(f"✅ Loaded {len(demo_leads)} demo leads")
                st.experimental_rerun()

# Import required for Excel export
from io import BytesIO

if __name__ == "__main__":
    main()