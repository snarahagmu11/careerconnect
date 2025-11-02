def split_text(text: str, chunk_size=800, overlap=120):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

