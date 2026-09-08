"""Shared course APIs; importing this module does not download a model."""
from .data import load_reviews
from .probability import filtered_distribution, distribution_table
from .model import LanguageModelLab, MODEL_ID, MODEL_REVISION

__version__ = "0.1.0"

def build_app(lab=None):
    """Build the cumulative Gradio app; None selects explicit numeric-only mode."""
    from .app import build_app as build
    return build(lab)

__all__ = ["LanguageModelLab", "MODEL_ID", "MODEL_REVISION", "load_reviews",
           "filtered_distribution", "distribution_table", "build_app"]
