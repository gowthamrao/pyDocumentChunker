import inspect
from tree_sitter_languages import get_parser, get_language

print(f"Is get_parser a class? {inspect.isclass(get_parser)}")
print(f"Is get_language a class? {inspect.isclass(get_language)}")

try:
    print("\nAttempting to get python parser...")
    parser = get_parser("python")
    print("Successfully got parser.")
    print("Parser language:", parser.language)
except Exception as e:
    print("Failed to get parser.")
    print(e)
