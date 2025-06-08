# Advanced Lead Intelligence Platform - Demo Notebook

## Overview
This notebook demonstrates the core functionality of the Lead Intelligence Platform, 
including lead generation, AI scoring, and segmentation capabilities.

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Lead Intelligence Platform Demo")
print("="*50)

## 1. Generate Sample Lead Data

def generate_sample_leads(n_samples=200):
    """Generate realistic sample lead data"""
    np.random.seed(42)
    
    # Company names and industries
    companies = []
    industries = ['Technology', 'Healthcare', 'Finance', 'Marketing', 'Manufacturing', 
                 'Education', 'Retail', 'Consulting', 'Real Estate', 'Media']
    
    company_prefixes = ['Tech', 'Digital', 'Smart', 'Advanced', 'Global', 'Innovative', 
                       'Future', 'Elite', 'Prime', 'Next']
    company_suffixes = ['Solutions', 'Systems', 'Corp', 'Inc', 'Group', 'Partners', 
                       'Enterprises', 'Technologies', 'Innovations', 'Consulting']
    
    for i in range(n_samples):
        company_name = f"{np.random.choice(company_prefixes)} {np.random.choice(company_suffixes)}"
        companies.append(company_name)
    
    # Generate lead data
    data = {
        'company_name': companies,
        'industry': np.random.choice(industries, n_samples),
        'company_size': np.random.choice(['1-10', '11-50', '51-200', '201-500', '501-1000'], 
                                       n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'location': np.random.choice(['California', 'New York', 'Texas', 'Florida', 'Illinois'], n_samples),
        'employees_on_linkedin': np.random.randint(5, 1000, n_samples),
        'follower_count': np.random.randint(100, 20000, n_samples),
        'has_email': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'has_website': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'has_phone': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'social_presence': np.random.choice(['Strong', 'Medium', 'Weak'], n_samples),
        'funding_stage': np.random.choice(['Seed', 'Series A', 'Series B', 'Established'], n_samples),
        'scraped_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    df = pd.DataFrame(data)
    
    # Generate lead scores based on features
    df['lead_score'] = (
        df['employees_on_linkedin'] * 0.05 +
        df['follower_count'] * 0.002 +
        df['has_email'] * 15 +
        df['has_website'] * 10 +
        df['has_phone'] * 8 +
        np.random.normal(0, 10, n_samples)
    ).clip(0, 100)
    
    return df

# Generate sample data
leads_df = generate_sample_leads(200)
print(f"✅ Generated {len(leads_df)} sample leads")
print(f"📊 Data shape: {leads_df.shape}")

# Display basic statistics
print("\n📈 Basic Statistics:")
print(leads_df.describe())

## 2. Lead Quality Prediction Model

class LeadQualityPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        features = df.copy()
        
        # Encode categorical variables
        if 'industry' not in self.label_encoders:
            self.label_encoders['industry'] = LabelEncoder()
            features['industry_encoded'] = self.label_encoders['industry'].fit_transform(features['industry'])
        else:
            features['industry_encoded'] = self.label_encoders['industry'].transform(features['industry'])
        
        # Company size mapping
        size_mapping = {'1-10': 1, '11-50': 2, '51-200': 3, '201-500': 4, '501-1000': 5}
        features['company_size_numeric'] = features['company_size'].map(size_mapping)
        
        # Log transformations
        features['employees_log'] = np.log1p(features['employees_on_linkedin'])
        features['followers_log'] = np.log1p(features['follower_count'])
        
        # Select features
        feature_columns = ['industry_encoded', 'company_size_numeric', 'employees_log', 
                          'followers_log', 'has_email', 'has_website', 'has_phone']
        
        self.feature_names = feature_columns
        return features[feature_columns]
    
    def train(self, df):
        """Train the lead quality prediction model"""
        features = self.prepare_features(df)
        
        # Create target variable (high quality leads)
        target = (df['lead_score'] > df['lead_score'].quantile(0.7)).astype(int)
        
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
        
        return {
            'accuracy': accuracy,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def predict(self, df):
        """Predict lead quality"""
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        predictions = self.model.predict(features_scaled)
        
        return probabilities, predictions

# Train lead quality model
print("\n🧠 Training Lead Quality Prediction Model...")
quality_predictor = LeadQualityPredictor()
training_results = quality_predictor.train(leads_df)

print(f"✅ Model trained successfully!")
print(f"📊 Accuracy: {training_results['accuracy']:.3f}")
print(f"🔢 Training samples: {training_results['n_train']}")
print(f"🔢 Test samples: {training_results['n_test']}")

# Feature importance
print("\n📈 Feature Importance:")
for feature, importance in sorted(training_results['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.3f}")

# Predict on full dataset
probabilities, predictions = quality_predictor.predict(leads_df)
leads_df['quality_probability'] = probabilities
leads_df['quality_prediction'] = predictions

print(f"\n🎯 High-quality leads identified: {predictions.sum()}/{len(predictions)}")

## 3. Lead Segmentation

class LeadSegmentation:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.scaler = StandardScaler()
    
    def segment_leads(self, df):
        """Segment leads using K-means clustering"""
        # Prepare features for clustering
        features = df[['lead_score', 'employees_on_linkedin', 'follower_count']].copy()
        
        # Log transform skewed features
        features['employees_log'] = np.log1p(features['employees_on_linkedin'])
        features['followers_log'] = np.log1p(features['follower_count'])
        
        # Normalize lead score
        features['lead_score_norm'] = features['lead_score'] / 100
        
        # Select final features
        cluster_features = features[['lead_score_norm', 'employees_log', 'followers_log']]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(cluster_features)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(features_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features_scaled, clusters)
        
        return clusters, silhouette_avg

# Perform lead segmentation
print("\n📊 Performing Lead Segmentation...")
segmentation = LeadSegmentation()
clusters, silhouette_score = segmentation.segment_leads(leads_df)

leads_df['segment'] = clusters
segment_names = {0: 'High Value', 1: 'Growth Potential', 2: 'Standard', 3: 'Low Priority'}
leads_df['segment_name'] = leads_df['segment'].map(segment_names)

print(f"✅ Segmentation complete!")
print(f"📊 Silhouette Score: {silhouette_score:.3f}")

# Segment statistics
print("\n📈 Segment Statistics:")
for segment_id, segment_name in segment_names.items():
    segment_data = leads_df[leads_df['segment'] == segment_id]
    print(f"\n{segment_name} (Segment {segment_id}):")
    print(f"  Count: {len(segment_data)}")
    print(f"  Avg Lead Score: {segment_data['lead_score'].mean():.1f}")
    print(f"  Avg Employees: {segment_data['employees_on_linkedin'].mean():.0f}")
    print(f"  Avg Followers: {segment_data['follower_count'].mean():.0f}")

## 4. Visualization and Analysis

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Lead Intelligence Platform - Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Lead Score Distribution
axes[0, 0].hist(leads_df['lead_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Lead Score Distribution')
axes[0, 0].set_xlabel('Lead Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(leads_df['lead_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {leads_df["lead_score"].mean():.1f}')
axes[0, 0].legend()

# 2. Industry Distribution
industry_counts = leads_df['industry'].value_counts()
axes[0, 1].bar(range(len(industry_counts)), industry_counts.values, color='lightcoral')
axes[0, 1].set_title('Industry Distribution')
axes[0, 1].set_xlabel('Industry')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_xticks(range(len(industry_counts)))
axes[0, 1].set_xticklabels(industry_counts.index, rotation=45, ha='right')

# 3. Company Size vs Lead Score
size_order = ['1-10', '11-50', '51-200', '201-500', '501-1000']
leads_df['company_size'] = pd.Categorical(leads_df['company_size'], categories=size_order, ordered=True)
size_scores = leads_df.groupby('company_size')['lead_score'].mean()
axes[1, 0].bar(range(len(size_scores)), size_scores.values, color='lightgreen')
axes[1, 0].set_title('Average Lead Score by Company Size')
axes[1, 0].set_xlabel('Company Size')
axes[1, 0].set_ylabel('Average Lead Score')
axes[1, 0].set_xticks(range(len(size_scores)))
axes[1, 0].set_xticklabels(size_scores.index, rotation=45)

# 4. Segment Distribution
segment_counts = leads_df['segment_name'].value_counts()
colors = ['gold', 'lightblue', 'lightcoral', 'lightgray']
axes[1, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
axes[1, 1].set_title('Lead Segment Distribution')

plt.tight_layout()
plt.show()

## 5. Export Results

# Create summary report
print("\n📋 LEAD INTELLIGENCE SUMMARY REPORT")
print("="*50)
print(f"📊 Total Leads Processed: {len(leads_df)}")
print(f"🎯 High-Quality Leads: {(leads_df['quality_prediction'] == 1).sum()}")
print(f"📈 Average Lead Score: {leads_df['lead_score'].mean():.1f}")
print(f"🏆 Top Industry: {leads_df['industry'].mode().iloc[0]}")
print(f"📍 Top Location: {leads_df['location'].mode().iloc[0]}")
print(f"🔍 Model Accuracy: {training_results['accuracy']:.3f}")
print(f"📊 Segmentation Quality: {silhouette_score:.3f}")

# Top performing leads
print(f"\n🌟 Top 10 Leads by Score:")
top_leads = leads_df.nlargest(10, 'lead_score')[['company_name', 'industry', 'lead_score', 'segment_name']]
for idx, row in top_leads.iterrows():
    print(f"  {row['company_name']} ({row['industry']}) - Score: {row['lead_score']:.1f} - {row['segment_name']}")

# Export data
output_file = f"lead_intelligence_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
leads_df.to_csv(output_file, index=False)
print(f"\n💾 Results exported to: {output_file}")

print("\n✅ Demo completed successfully!")
print("🚀 Ready for production deployment!")