import google.generativeai as genai
import os
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
for m in genai.list_models():
    if "audio" in str(m.supported_generation_methods) or "generateContent" in m.supported_generation_methods:
        print(m.name, m.supported_generation_methods)
    

    
