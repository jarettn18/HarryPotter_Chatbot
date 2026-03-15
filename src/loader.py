from dataclasses import dataclass, field
from datasets import load_dataset


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def load_harry_potter_dataset() -> list[Document]:
    """Load Harry Potter text from HuggingFace and return as Document objects."""
    dataset = load_dataset("elricwan/HarryPotter", split="train")

    documents = []
    for i, row in enumerate(dataset):
        text = row.get("content", "")
        if not text or not text.strip():
            continue

        metadata = {
            "source": "harry_potter",
            "index": i,
        }
        # Include any other fields from the dataset as metadata
        for key, value in row.items():
            if key != "text" and isinstance(value, (str, int, float)):
                metadata[key] = value

        documents.append(Document(text=text.strip(), metadata=metadata))

    print(f"Loaded {len(documents)} documents from Harry Potter dataset")
    return documents
