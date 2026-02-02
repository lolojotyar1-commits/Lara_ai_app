import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# إعدادات الصفحة
st.set_page_config(page_title="Lara AI Study", page_icon="🎓", layout="wide")

# تصميم الواجهة
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    * { font-family: 'Cairo', sans-serif; direction: rtl; }
    .stApp { background-color: #ffffff; }
    .main-title { color: #6c5ce7; text-align: center; font-size: 3rem; font-weight: bold; }
    .footer { position: fixed; bottom: 0; left: 0; width: 100%; background: #6c5ce7; color: white; text-align: center; padding: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📚 منصة لارا التعليمية الذكية</p>', unsafe_allow_html=True)

# جلب المفتاح من Secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ يرجى ضبط الـ API Key في إعدادات Streamlit Secrets")
    st.stop()

# رفع الملفات
uploaded_files = st.file_uploader("📂 ارفعي كتب المنهج (PDF)", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 تحليل الكتب وبدء الدراسة"):
        with st.spinner("جاري معالجة الكتاب... لارا تعمل بجد 👩‍💻"):
            text = ""
            for pdf in uploaded_files:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            st.session_state.vectors = vector_store
            st.success("تم تحليل الكتاب بنجاح! يمكنج السؤال الآن.")

# الشات
query = st.text_input("❓ ما هو سؤالك من داخل الكتاب؟")
if query and "vectors" in st.session_state:
    with st.spinner("جاري استخراج الإجابة من المنهج..."):
        llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")
        template = """أنت معلم مساعد خبير. استخدم النص المرفق فقط للإجابة على السؤال. 
        إذا لم تجد الإجابة، قل أنها غير موجودة في الكتاب المرفق.
        النص: {context}
        السؤال: {question}
        الإجابة:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vectors.as_retriever(), chain_type_kwargs={"prompt": prompt})
        
        response = chain.run(query)
        st.info(response)

# الفوتر الثابت
st.markdown('<div class="footer">صنع بكل حب بواسطة المطورة لارا ❤️ 2026</div>', unsafe_allow_html=True)
