"""
FinBERT Sentiment Analyzer for Financial News.

Uses the ProsusAI/finbert model to analyze sentiment of financial news headlines.
Supports batch processing with caching for efficiency.

Usage:
    from src.data.news import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("Apple stock rises on strong earnings")
    print(result.sentiment_label, result.sentiment_score)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import warnings

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')


@dataclass
class SentimentScore:
    """Result of sentiment analysis for a single text."""
    text: str
    sentiment_label: str      # 'positive', 'negative', 'neutral'
    sentiment_score: float    # -1 to +1 (negative to positive)
    positive_prob: float      # 0 to 1
    negative_prob: float      # 0 to 1
    neutral_prob: float       # 0 to 1
    confidence: float         # Max of the three probabilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'sentiment_label': self.sentiment_label,
            'sentiment_score': self.sentiment_score,
            'sentiment_positive': self.positive_prob,
            'sentiment_negative': self.negative_prob,
            'sentiment_neutral': self.neutral_prob,
            'confidence': self.confidence,
        }


class SentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial news.

    Uses the ProsusAI/finbert model which is specifically trained on
    financial text for more accurate sentiment classification.

    Args:
        model_name: HuggingFace model name (default: ProsusAI/finbert)
        device: Device to use ('cuda', 'cpu', or 'auto')
        batch_size: Batch size for inference
        max_length: Maximum token length for texts

    Example:
        analyzer = SentimentAnalyzer()

        # Single text
        result = analyzer.analyze("Apple beats earnings expectations")
        print(result.sentiment_label)  # 'positive'

        # Batch processing
        texts = ["Stock falls", "Revenue grows", "Market stable"]
        results = analyzer.analyze_batch(texts)
    """

    # Class-level model cache to avoid reloading
    _model = None
    _tokenizer = None
    _device = None

    def __init__(
        self,
        model_name: str = 'ProsusAI/finbert',
        device: str = 'auto',
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._requested_device = device

        # Lazy load - don't initialize until first use
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Determine device
            if self._requested_device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                    logger.info("Using CUDA for sentiment analysis")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                    logger.info("Using MPS (Apple Silicon) for sentiment analysis")
                else:
                    device = 'cpu'
                    logger.info("Using CPU for sentiment analysis")
            else:
                device = self._requested_device

            # Use class-level cache
            if SentimentAnalyzer._model is None:
                logger.info(f"Loading sentiment model: {self.model_name}")
                SentimentAnalyzer._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                SentimentAnalyzer._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                SentimentAnalyzer._device = device
                SentimentAnalyzer._model.to(device)
                SentimentAnalyzer._model.eval()
                logger.info("Sentiment model loaded successfully")

            self._initialized = True

        except ImportError as e:
            raise ImportError(
                "Sentiment analysis requires 'transformers' and 'torch' packages. "
                "Install with: pip install transformers torch"
            ) from e

    @property
    def model(self):
        """Get the loaded model."""
        self._ensure_initialized()
        return SentimentAnalyzer._model

    @property
    def tokenizer(self):
        """Get the loaded tokenizer."""
        self._ensure_initialized()
        return SentimentAnalyzer._tokenizer

    @property
    def device(self):
        """Get the device being used."""
        self._ensure_initialized()
        return SentimentAnalyzer._device

    def analyze(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze (headline or summary)

        Returns:
            SentimentScore with label, score, and probabilities
        """
        results = self.analyze_batch([text])
        return results[0]

    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """
        Analyze sentiment of multiple texts in batches.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentScore objects
        """
        import torch

        self._ensure_initialized()

        if not texts:
            return []

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            # Process results
            # FinBERT labels: 0=positive, 1=negative, 2=neutral
            for j, text in enumerate(batch_texts):
                pos_prob = float(probs[j, 0])
                neg_prob = float(probs[j, 1])
                neu_prob = float(probs[j, 2])

                # Determine label
                max_idx = int(np.argmax(probs[j]))
                labels = ['positive', 'negative', 'neutral']
                label = labels[max_idx]

                # Compute sentiment score: positive - negative
                # Range: -1 (fully negative) to +1 (fully positive)
                sentiment_score = pos_prob - neg_prob

                # Confidence is the max probability
                confidence = float(np.max(probs[j]))

                results.append(SentimentScore(
                    text=text,
                    sentiment_label=label,
                    sentiment_score=sentiment_score,
                    positive_prob=pos_prob,
                    negative_prob=neg_prob,
                    neutral_prob=neu_prob,
                    confidence=confidence,
                ))

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'headline',
        prefix: str = 'sentiment_',
    ) -> pd.DataFrame:
        """
        Add sentiment columns to a DataFrame.

        Args:
            df: DataFrame with text column
            text_column: Column containing text to analyze
            prefix: Prefix for sentiment columns

        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty:
            return df

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        texts = df[text_column].fillna('').tolist()
        results = self.analyze_batch(texts)

        # Add columns
        df = df.copy()
        df[f'{prefix}score'] = [r.sentiment_score for r in results]
        df[f'{prefix}label'] = [r.sentiment_label for r in results]
        df[f'{prefix}positive'] = [r.positive_prob for r in results]
        df[f'{prefix}negative'] = [r.negative_prob for r in results]
        df[f'{prefix}neutral'] = [r.neutral_prob for r in results]
        df[f'{prefix}confidence'] = [r.confidence for r in results]

        return df


class MockSentimentAnalyzer:
    """
    Mock sentiment analyzer for testing without loading the model.

    Returns random sentiment scores that can be seeded for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def analyze(self, text: str) -> SentimentScore:
        """Generate mock sentiment for a single text."""
        results = self.analyze_batch([text])
        return results[0]

    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Generate mock sentiment for multiple texts."""
        results = []
        for text in texts:
            # Generate random probabilities
            probs = self.rng.dirichlet([1, 1, 1])
            pos_prob, neg_prob, neu_prob = probs

            # Determine label
            labels = ['positive', 'negative', 'neutral']
            label = labels[int(np.argmax(probs))]

            results.append(SentimentScore(
                text=text,
                sentiment_label=label,
                sentiment_score=pos_prob - neg_prob,
                positive_prob=float(pos_prob),
                negative_prob=float(neg_prob),
                neutral_prob=float(neu_prob),
                confidence=float(np.max(probs)),
            ))

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'headline',
        prefix: str = 'sentiment_',
    ) -> pd.DataFrame:
        """Add mock sentiment columns to a DataFrame."""
        if df.empty:
            return df

        texts = df[text_column].fillna('').tolist()
        results = self.analyze_batch(texts)

        df = df.copy()
        df[f'{prefix}score'] = [r.sentiment_score for r in results]
        df[f'{prefix}label'] = [r.sentiment_label for r in results]
        df[f'{prefix}positive'] = [r.positive_prob for r in results]
        df[f'{prefix}negative'] = [r.negative_prob for r in results]
        df[f'{prefix}neutral'] = [r.neutral_prob for r in results]
        df[f'{prefix}confidence'] = [r.confidence for r in results]

        return df


def get_sentiment_analyzer(use_mock: bool = False, **kwargs) -> SentimentAnalyzer:
    """
    Factory function to get a sentiment analyzer.

    Args:
        use_mock: If True, return MockSentimentAnalyzer for testing
        **kwargs: Arguments passed to SentimentAnalyzer

    Returns:
        SentimentAnalyzer or MockSentimentAnalyzer instance
    """
    if use_mock:
        seed = kwargs.pop('seed', None)
        return MockSentimentAnalyzer(seed=seed)
    return SentimentAnalyzer(**kwargs)
