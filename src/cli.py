import argparse
import sys

from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Harry Potter RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("ingest", help="Ingest the Harry Potter dataset into ChromaDB")

    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A chat")
    chat_parser.add_argument("--show-sources", action="store_true", help="Show retrieved source passages")

    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question", type=str, help="The question to ask")
    query_parser.add_argument("--show-sources", action="store_true", help="Show retrieved source passages")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    pipeline = RAGPipeline()

    if args.command == "ingest":
        pipeline.ingest()

    elif args.command == "query":
        answer, sources = pipeline.query_with_sources(args.question)
        print(f"\nAnswer:\n{answer}")
        if args.show_sources:
            print_sources(sources)

    elif args.command == "chat":
        print("Harry Potter RAG Chat (type 'quit' to exit)\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            answer, sources = pipeline.query_with_sources(question)
            print(f"\nAssistant: {answer}\n")
            if args.show_sources:
                print_sources(sources)


def print_sources(sources: list[dict]) -> None:
    """Print retrieved source passages."""
    print("\n--- Retrieved Sources ---")
    for i, source in enumerate(sources, 1):
        text_preview = source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"]
        print(f"\n[{i}] {text_preview}")
    print("--- End Sources ---\n")


if __name__ == "__main__":
    main()
