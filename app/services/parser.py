import os
import pandas as pd
import fitz
import pdfplumber


def read_txt(file_path:str)->str:
    with open(file_path,'r',encoding="utf-8") as file:
        content=file.read()
        return content
    
def read_csv(file_path:str)->pd.DataFrame:

    df=pd.read_csv(file_path,encoding='utf-8')
    return  "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))

def read_pdf(file_path:str)->str:
    text=""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text+= page.extract_text()+"\n"

    return text.strip()

def load_document(file_path:str)->str:
    ext=os.path.splitext(file_path)[1].lower()

    if ext==".txt":
        return read_txt(file_path)
    
    elif ext==".csv":
        return read_csv(file_path)
    
    elif ext==".pdf":   
        return read_pdf(file_path)
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")