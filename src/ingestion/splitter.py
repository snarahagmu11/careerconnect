def split_text(text: str, chunk_size=600, chunk_overlap=80):
    chunks = []
    i = 0
    step = max(chunk_size - chunk_overlap, 1)
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += step
    return chunks

def split_documents(texts, chunk_size=600, chunk_overlap=80):
    out = []
    for t in texts:
        if t:
            out.extend(split_text(str(t), chunk_size, chunk_overlap))
    return out

