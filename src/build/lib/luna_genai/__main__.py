"""Run: python -m luna_genai [--numeric-only]. Local host by default."""
import argparse

def main():
    from . import LanguageModelLab, build_app
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-only", action="store_true")
    args = parser.parse_args()
    lab = None if args.numeric_only else LanguageModelLab()
    build_app(lab).launch(server_name="127.0.0.1", share=False)

if __name__ == "__main__":
    main()
