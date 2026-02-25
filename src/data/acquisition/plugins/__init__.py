"""Plugin registry for data acquisition sources."""

PLUGIN_REGISTRY = {
    "equities": "src.data.acquisition.plugins.alpaca_equities.AlpacaEquitiesPlugin",
    "crypto": "src.data.acquisition.plugins.alpaca_crypto.AlpacaCryptoPlugin",
    "futures": "src.data.acquisition.plugins.databento_futures.DatabentoFuturesPlugin",
    "news": "src.data.acquisition.plugins.alpaca_news.AlpacaNewsPlugin",
}


def get_plugin_class(source: str):
    """Lazy-load and return a plugin class by source name."""
    if source not in PLUGIN_REGISTRY:
        raise ValueError(
            f"Unknown source '{source}'. Available: {list(PLUGIN_REGISTRY.keys())}"
        )
    module_path, class_name = PLUGIN_REGISTRY[source].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
