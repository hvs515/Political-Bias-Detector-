"""
Comprehensive feature engineering for political bias detection.
Builds 6 types of features:
1. TF-IDF features
2. Political lexicon features
3. Sentiment features
4. Stylistic features
5. Named entity features
6. Readability features
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import textstat
import warnings

warnings.filterwarnings('ignore')


class PoliticalBiasFeatureExtractor:
    """Main feature extractor combining all features."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = MaxAbsScaler()
        self.nlp = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self._load_spacy_model()
    
    def __getstate__(self):
        """Custom pickle state to handle scaler properly."""
        state = self.__dict__.copy()
        # Save scaler parameters if fitted
        if hasattr(self.scaler, 'scale_'):
            state['scaler_params'] = {
                'scale_': self.scaler.scale_,
                'max_abs_': getattr(self.scaler, 'max_abs_', None),
                'n_features_in_': getattr(self.scaler, 'n_features_in_', None),
                'feature_names_in_': getattr(self.scaler, 'feature_names_in_', None),
            }
        else:
            state['scaler_params'] = None
        # Don't pickle the scaler object itself
        state['scaler'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickle state to restore scaler."""
        scaler_params = state.pop('scaler_params', None)
        self.__dict__.update(state)
        # Reinitialize the scaler
        self.scaler = MaxAbsScaler()
        # Restore fitted parameters if they exist
        if scaler_params:
            for attr, value in scaler_params.items():
                if value is not None:
                    setattr(self.scaler, attr, value)
    
    def _load_spacy_model(self):
        """Load spacy model, download if needed."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model en_core_web_sm...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                          capture_output=True)
            self.nlp = spacy.load("en_core_web_sm")
    
    # ==================== 1. TF-IDF Features ====================
    
    def extract_tfidf_features(self, texts, fit=False):
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts: list of text strings
            fit: if True, fit the vectorizer on texts
            
        Returns:
            scipy.sparse matrix of TF-IDF features
        """
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,  # Reduced for synthetic dataset
                sublinear_tf=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.85,
                stop_words='english'
            )
            return self.tfidf_vectorizer.fit_transform(texts)
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("Vectorizer not fitted. Call with fit=True first.")
            return self.tfidf_vectorizer.transform(texts)
    
    # ==================== 2. Political Lexicon Features ====================
    
    def extract_lexicon_features(self, texts):
        """
        Extract political lexicon features.
        
        Returns:
            DataFrame with columns: left_count, right_count, loaded_count,
            left_ratio, right_ratio, loaded_ratio, net_lean_score
        """
        left_lexicon = {
            'undocumented', 'progressive', 'systemic', 'marginalized',
            'equity', 'climate crisis', 'living wage', 'gun control', 
            'reproductive rights', 'environmental justice', 'social justice',
            'wealth inequality', 'police brutality', 'immigration reform',
            'healthcare for all', 'green energy', 'civil rights', 'unions'
        }
        
        right_lexicon = {
            'illegal alien', 'illegal aliens', 'radical left', 'deep state', 
            'job creator', 'job creators', 'second amendment', 'free market', 
            'big government', 'election integrity', 'constitutional rights',
            'traditional values', 'law and order', 'border security',
            'socialism', 'cancel culture', 'liberal media', 'government overreach'
        }
        
        loaded_words = {
            'regime', 'radical', 'extreme', 'dangerous', 'corrupt',
            'rigged', 'stolen', 'crisis', 'invasion', 'threat',
            'destroy', 'save', 'patriots', 'treason', 'un-American'
        }
        
        features = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Count occurrences
            left_count = sum(1 for term in left_lexicon if term in text_lower)
            right_count = sum(1 for term in right_lexicon if term in text_lower)
            loaded_count = sum(1 for term in loaded_words if term in text_lower)
            
            # Word count for normalization
            word_count = len(text_lower.split())
            word_count = max(word_count, 1)  # Avoid division by zero
            
            # Ratios
            left_ratio = left_count / word_count
            right_ratio = right_count / word_count
            loaded_ratio = loaded_count / word_count
            
            # Net lean score: positive = right lean, negative = left lean
            net_lean_score = right_count - left_count
            
            features.append({
                'left_count': left_count,
                'right_count': right_count,
                'loaded_count': loaded_count,
                'left_ratio': left_ratio,
                'right_ratio': right_ratio,
                'loaded_ratio': loaded_ratio,
                'net_lean_score': net_lean_score
            })
        
        return pd.DataFrame(features)
    
    # ==================== 3. Sentiment Features ====================
    
    def extract_sentiment_features(self, texts):
        """
        Extract sentiment features using VADER.
        
        Returns:
            DataFrame with columns: sentiment_pos, sentiment_neg, sentiment_neu,
            sentiment_compound, sentiment_variance, sentiment_range
        """
        features = []
        
        for text in texts:
            try:
                # Simple sentence splitting using periods
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            except:
                sentences = [text]
            
            sentence_sentiments = []
            overall_sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            for sent in sentences:
                if sent.strip():
                    sent_sentiment = self.sentiment_analyzer.polarity_scores(sent)
                    sentence_sentiments.append(sent_sentiment['compound'])
            
            # Sentiment variance and range
            if sentence_sentiments:
                sentiment_variance = np.var(sentence_sentiments)
                sentiment_range = max(sentence_sentiments) - min(sentence_sentiments)
            else:
                sentiment_variance = 0
                sentiment_range = 0
            
            features.append({
                'sentiment_pos': overall_sentiment['pos'],
                'sentiment_neg': overall_sentiment['neg'],
                'sentiment_neu': overall_sentiment['neu'],
                'sentiment_compound': overall_sentiment['compound'],
                'sentiment_variance': sentiment_variance,
                'sentiment_range': sentiment_range
            })
        
        return pd.DataFrame(features)
    
    # ==================== 4. Stylistic Features ====================
    
    def extract_stylistic_features(self, texts):
        """
        Extract stylistic features.
        
        Returns:
            DataFrame with multiple stylistic feature columns
        """
        hedge_words = {'allegedly', 'reportedly', 'claims', 'suggests', 'appears',
                      'indicates', 'potentially', 'possibly', 'arguably'}
        
        certainty_words = {'definitely', 'clearly', 'obviously', 'always', 'never',
                          'absolutely', 'certainly', 'undoubtedly', 'unquestionably'}
        
        first_person = {'we', 'our', 'us', 'i', 'me', 'my', 'mine'}
        third_person = {'they', 'them', 'their', 'theirs', 'he', 'she', 'it', 'its'}
        
        features = []
        
        for text in texts:
            # Simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Simple word tokenization (split on whitespace and remove punctuation)
            words = []
            for word in text.lower().split():
                # Remove basic punctuation
                word = ''.join(c for c in word if c.isalnum())
                if word:
                    words.append(word)
            
            word_count = len(words)
            
            # Average sentence length
            avg_sent_length = word_count / max(len(sentences), 1)
            
            # Average word length
            avg_word_length = np.mean([len(w) for w in words if w.isalnum()]) if words else 0
            
            # Hedge words
            hedge_count = sum(1 for w in words if w in hedge_words)
            hedge_ratio = hedge_count / max(word_count, 1)
            
            # Certainty words
            certainty_count = sum(1 for w in words if w in certainty_words)
            certainty_ratio = certainty_count / max(word_count, 1)
            
            # First person
            first_person_count = sum(1 for w in words if w in first_person)
            first_person_ratio = first_person_count / max(word_count, 1)
            
            # Third person
            third_person_count = sum(1 for w in words if w in third_person)
            third_person_ratio = third_person_count / max(word_count, 1)
            
            # Punctuation
            exclamation_ratio = text.count('!') / max(len(sentences), 1)
            question_ratio = text.count('?') / max(len(sentences), 1)
            quote_ratio = text.count('"') / max(word_count, 1)
            
            # Certainty to hedge ratio
            certainty_to_hedge_ratio = certainty_count / max(hedge_count, 1)
            
            features.append({
                'avg_sentence_length': avg_sent_length,
                'avg_word_length': avg_word_length,
                'hedge_ratio': hedge_ratio,
                'certainty_ratio': certainty_ratio,
                'first_person_ratio': first_person_ratio,
                'third_person_ratio': third_person_ratio,
                'exclamation_ratio': exclamation_ratio,
                'question_ratio': question_ratio,
                'quote_ratio': quote_ratio,
                'certainty_to_hedge': certainty_to_hedge_ratio
            })
        
        return pd.DataFrame(features)
    
    # ==================== 5. Named Entity Features ====================
    
    def extract_entity_features(self, texts):
        """
        Extract named entity features using spacy.
        
        Returns:
            DataFrame with entity-based features
        """
        left_entities = {'biden', 'obama', 'cnn', 'msnbc', 'new york times',
                        'washington post', 'bernie', 'aoc', 'squad'}
        
        right_entities = {'trump', 'fox', 'breitbart', 'hannity', 'desantis',
                         'mtg', 'boebert', 'newsmax', 'oan'}
        
        features = []
        
        for text in texts:
            try:
                doc = self.nlp(text[:1000000])  # Limit to avoid processing huge texts
            except:
                doc = None
            
            # Count entity types
            person_count = 0
            org_count = 0
            gpe_count = 0
            
            left_entity_hits = 0
            right_entity_hits = 0
            
            if doc:
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        person_count += 1
                        if ent.text.lower() in left_entities:
                            left_entity_hits += 1
                        if ent.text.lower() in right_entities:
                            right_entity_hits += 1
                    elif ent.label_ == 'ORG':
                        org_count += 1
                        if ent.text.lower() in left_entities:
                            left_entity_hits += 1
                        if ent.text.lower() in right_entities:
                            right_entity_hits += 1
                    elif ent.label_ == 'GPE':
                        gpe_count += 1
            
            word_count = len(text.split())
            word_count = max(word_count, 1)
            
            # Entity lean score
            entity_lean_score = right_entity_hits - left_entity_hits
            
            features.append({
                'person_count': person_count / word_count,
                'org_count': org_count / word_count,
                'gpe_count': gpe_count / word_count,
                'left_entity_hits': left_entity_hits,
                'right_entity_hits': right_entity_hits,
                'entity_lean_score': entity_lean_score
            })
        
        return pd.DataFrame(features)
    
    # ==================== 6. Readability Features ====================
    
    def extract_readability_features(self, texts):
        """
        Extract readability features using textstat.
        
        Returns:
            DataFrame with readability scores
        """
        features = []
        
        for text in texts:
            try:
                flesch_ease = textstat.flesch_reading_ease(text)
            except:
                flesch_ease = 0
            
            try:
                flesch_kincaid = textstat.flesch_kincaid_grade(text)
            except:
                flesch_kincaid = 0
            
            try:
                gunning_fog = textstat.gunning_fog(text)
            except:
                gunning_fog = 0
            
            try:
                smog = textstat.smog_index(text)
            except:
                smog = 0
            
            try:
                ari = textstat.automated_readability_index(text)
            except:
                ari = 0
            
            features.append({
                'flesch_reading_ease': flesch_ease,
                'flesch_kincaid_grade': flesch_kincaid,
                'gunning_fog': gunning_fog,
                'smog_index': smog,
                'automated_readability_index': ari
            })
        
        return pd.DataFrame(features)
    
    # ==================== Main Feature Extraction ====================
    
    def extract_all_features(self, texts, fit_tfidf=False):
        """
        Extract all features and combine them using scipy sparse hstack.
        
        Args:
            texts: list of text strings
            fit_tfidf: if True, fit TF-IDF vectorizer on texts
            
        Returns:
            scipy.sparse matrix of combined features
        """
        print("Extracting TF-IDF features...")
        tfidf_features = self.extract_tfidf_features(texts, fit=fit_tfidf)
        
        print("Extracting lexicon features...")
        lexicon_features = self.extract_lexicon_features(texts)
        
        print("Extracting sentiment features...")
        sentiment_features = self.extract_sentiment_features(texts)
        
        print("Extracting stylistic features...")
        stylistic_features = self.extract_stylistic_features(texts)
        
        print("Extracting entity features...")
        entity_features = self.extract_entity_features(texts)
        
        print("Extracting readability features...")
        readability_features = self.extract_readability_features(texts)
        
        # Combine hand-crafted features
        hand_crafted = pd.concat([
            lexicon_features,
            sentiment_features,
            stylistic_features,
            entity_features,
            readability_features
        ], axis=1)
        
        # Scale hand-crafted features
        hand_crafted_scaled = self.scaler.fit_transform(hand_crafted) if fit_tfidf else \
                              self.scaler.transform(hand_crafted)
        hand_crafted_sparse = csr_matrix(hand_crafted_scaled)
        
        # Combine all features
        combined_features = hstack([tfidf_features, hand_crafted_sparse])
        
        print(f"Total features: {combined_features.shape[1]}")
        
        return combined_features, hand_crafted
    
    def get_feature_names(self):
        """Get names of all TF-IDF features."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted.")
        return self.tfidf_vectorizer.get_feature_names_out()


if __name__ == "__main__":
    # Test feature extraction
    from preprocess import load_or_create_dataset
    
    print("Loading dataset...")
    df = load_or_create_dataset()
    texts = df['text'].tolist()[:10]  # Test on first 10 texts
    
    print("Initializing feature extractor...")
    extractor = PoliticalBiasFeatureExtractor()
    
    print("Extracting features...")
    features, hand_crafted = extractor.extract_all_features(texts, fit_tfidf=True)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Hand-crafted features shape: {hand_crafted.shape}")
    print("\nHand-crafted features sample:")
    print(hand_crafted.head())
