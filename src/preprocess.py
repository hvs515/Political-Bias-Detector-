"""
Data preprocessing and synthetic dataset generation for political bias detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


def generate_synthetic_dataset(n_samples=500, random_state=42):
    """
    Generate synthetic dataset with political bias labels.
    100 samples per class: FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT
    """
    np.random.seed(random_state)
    
    classes = ['FAR_LEFT', 'LEFT', 'CENTER', 'RIGHT', 'FAR_RIGHT']
    n_per_class = n_samples // len(classes)
    
    far_left_keywords = [
        'systemic racism', 'progressive reform', 'climate crisis', 'marginalized communities',
        'equity', 'undocumented', 'gun control', 'reproductive rights', 'living wage',
        'corporate greed', 'environmental justice', 'social justice', 'wealth inequality',
        'police brutality', 'immigration reform', 'healthcare for all',
    ]
    
    left_keywords = [
        'progressive', 'regulation', 'climate change', 'healthcare reform', 'workers rights',
        'income inequality', 'gun safety', 'voting rights', 'environmental protection',
        'tax the rich', 'free college', 'green energy', 'unions', 'civil rights',
    ]
    
    center_keywords = [
        'congress', 'legislation', 'report shows', 'according to', 'studies suggest',
        'analysis reveals', 'officials say', 'government', 'economy', 'policy',
        'approach', 'data', 'research', 'moderate', 'balance', 'context',
    ]
    
    right_keywords = [
        'illegal aliens', 'radical left', 'deep state', 'job creators', 'second amendment',
        'free market', 'big government', 'election integrity', 'constitutional rights',
        'traditional values', 'law and order', 'border security', 'socialism',
        'cancel culture', 'liberal media', 'government overreach',
    ]
    
    far_right_keywords = [
        'invasion', 'radical extremists', 'destroy America', 'patriots', 'regime',
        'dangerous criminals', 'stolen election', 'rigged system', 'survive',
        'forced', 'danger', 'threat to our way', 'un-American', 'treason',
        'save the nation', 'stand up now',
    ]
    
    template_topics = [
        'The government announced a new policy on immigration.',
        'Recent reports discuss changes in tax policy.',
        'Officials comment on environmental regulations.',
        'Analysis of recent election results.',
        'Discussion of healthcare system changes.',
        'Commentary on international relations.',
        'Economic trends and employment figures.',
        'Education policy debate continues.',
        'Discussion of law enforcement practices.',
        'Climate and energy policy changes.',
    ]
    
    data = []
    
    for class_idx, class_label in enumerate(classes):
        if class_label == 'FAR_LEFT':
            keywords = far_left_keywords
        elif class_label == 'LEFT':
            keywords = left_keywords
        elif class_label == 'CENTER':
            keywords = center_keywords
        elif class_label == 'RIGHT':
            keywords = right_keywords
        else:
            keywords = far_right_keywords
        
        for i in range(n_per_class):
            topic = np.random.choice(template_topics)
            
            # Create article text with biased keywords
            num_keywords = np.random.randint(4, 8)
            selected_keywords = np.random.choice(keywords, num_keywords, replace=False)
            
            article_parts = [topic]
            article_parts.extend(selected_keywords)
            
            # Add more filler text
            filler = "The situation continues to develop as stakeholders discuss implications. "
            filler += "Expert analysis suggests various perspectives on the issue. "
            filler += "More information is expected to emerge in coming days."
            
            article_text = ". ".join(article_parts) + ". " + filler
            
            data.append({
                'text': article_text,
                'bias': class_label
            })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def load_or_create_dataset(data_path='data/raw/articles.csv', create_if_missing=True):
    """
    Load dataset from CSV or create synthetic if missing.
    """
    data_path = Path(data_path)
    
    if data_path.exists():
        print(f"Loading dataset from {data_path}")
        return pd.read_csv(data_path)
    elif create_if_missing:
        print("Creating synthetic dataset...")
        df = generate_synthetic_dataset(n_samples=500)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Synthetic dataset saved to {data_path}")
        return df
    else:
        raise FileNotFoundError(f"Dataset not found at {data_path}")


if __name__ == "__main__":
    # Generate and save synthetic dataset
    df = generate_synthetic_dataset(n_samples=500)
    output_path = Path("data/raw/articles.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset created: {output_path}")
    print(f"Shape: {df.shape}")
    print(df['bias'].value_counts())
