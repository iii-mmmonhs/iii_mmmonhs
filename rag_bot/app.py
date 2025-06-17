from fastapi import FastAPI
import gradio as gr
from config import PDF_PATH
from utils.pdf_parser import extract_text_from_pdf, split_text_into_chunks
from utils.vectorstore import build_vectorstore, load_vectorstore, retrieve_relevant_chunks
from utils.rag_pipeline import generate_answer

print("Подготовка данных...")
raw_text = extract_text_from_pdf(PDF_PATH)
chunks = split_text_into_chunks(raw_text, chunk_size=512, overlap=64)
index, chunks, model = build_vectorstore(model_name="all-MiniLM-L6-v2", chunks=chunks)

def query_rag(question: str):
    relevant = retrieve_relevant_chunks(question, index, chunks, model)
    context = "\n\n".join(relevant[:2])
    answer = generate_answer(context, question)
    return answer

app = FastAPI()

@app.get("/query")
def query(question: str):
    return {"answer": query_rag(question)}

demo = gr.Interface(fn=query_rag, inputs="text", outputs="text")
app.mount("/gradio", gr.routes.App(demo, None, None))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)