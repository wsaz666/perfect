from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def semantic_chunk(text:str,file_ext:str,chunk_size=800,chunkoverlap=20) -> List[dict]:
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunkoverlap,
        separators=["\n\n","。","，","！","？", "；", "\n", " ", ""],
        keep_separators=True
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunkoverlap,
        separators=["<|endoftext|>","\n\n", "\n", " ", ""],
        keep_separators=True
    )
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunkoverlap,
        separators=["endmodule","\n\n", "\n", " ", ""],
        keep_separators=True
    )
    file_ext=file_ext.lower()
    chunks=[]
    if file_ext==".pdf":
        chunks = pdf_splitter.split_text(text)
    elif file_ext==".txt":
        chunks = text_splitter.split_text(text)
    elif file_ext in [".v",".sv"]:
        chunks = code_splitter.split_text(text)
    else:
        chunks = pdf_splitter.split_text(text)

    chunk_data_list = []
    for i,chunk in enumerate(chunks):
        if len(chunk)<20: continue
        chunk_data_list.append({
                "id": f'chunk{i}',
                "chunk": chunk,
                "method": "semantic_chunk"
            })
    return chunk_data_list