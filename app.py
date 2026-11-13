import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INNER_DIR = os.path.join(BASE_DIR, "fake_news_classifier")
if INNER_DIR not in sys.path:
    sys.path.insert(0, INNER_DIR)

import importlib.util

spec = importlib.util.spec_from_file_location("classifier_app", os.path.join(INNER_DIR, "app.py"))
classifier_module = importlib.util.module_from_spec(spec)
sys.modules["classifier_app"] = classifier_module
spec.loader.exec_module(classifier_module)
app = classifier_module.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
