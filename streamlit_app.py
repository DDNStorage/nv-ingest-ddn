import logging
import os
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import psutil
import streamlit as st

from nv_ingest_client.util.milvus import nvingest_retrieval
from nv_ingest_client.client import Ingestor

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Performance metrics functions
def start_performance_metrics() -> Tuple[float, int, float]:
    step_start_time = time.time()
    memory_before, _ = tracemalloc.get_traced_memory()
    cpu_before = psutil.cpu_percent(interval=0.1)
    return step_start_time, memory_before, cpu_before


def end_performance_metrics(
    step_start_time: float, memory_before: int, cpu_before: float, text: str
) -> None:
    cpu_after = psutil.cpu_percent(interval=None)
    memory_after, _ = tracemalloc.get_traced_memory()
    step_end_time = time.time()

    elapsed_time = step_end_time - step_start_time
    memory_diff = (memory_after - memory_before) / (1024 * 1024)
    cpu_usage = cpu_after

    # Log to console/file using logger
    logger.info(
        f"[{text}] Time: {elapsed_time:.4f}s, "
        f"Memory Diff: {memory_diff:.4f} MB, "
        f"CPU usage diff: {cpu_usage:.2f}%"
    )


###########################################################
##################### MULTIMODAL-PART #####################
###########################################################


def multimodal_ingestion(pdf_path):

    logger.info(f"PDF path: {pdf_path}")
    logger.info(
        f"EMbedding path: {os.environ.get('MULTIMODAL_EMBEDDING_ENDPOINT', 'http://localhost:8000')}."
    )
    try:
        ingestor = (
            Ingestor(message_client_hostname="localhost")
            .files(pdf_path)
            .extract(
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_images=True,
                text_depth="page",
            )
            .dedup(
                content_type="image",
                filter=True,
            )
            .filter(
                content_type="image",
                min_size=128,
                max_aspect_ratio=5.0,
                min_aspect_ratio=0.2,
                filter=True,
            )
            .split(
                tokenizer="meta-llama/Llama-3.2-1B",
                chunk_size=1024,
                chunk_overlap=150,
            )
            .embed()
            .vdb_upload(dense_dim=2048)
        )
        response = ingestor.ingest()[0]
    except Exception as e:
        logger.error(f"Multimodel Ingestion FAILED: {e}.")
    return response


def multimodal_query(query):
    results = nvingest_retrieval(
        [query],
        "nv_ingest_collection",
        hybrid=False,
        embedding_endpoint="http://localhost:8012/v1",
        model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
        top_k=1,
        gpu_search=True,
    )
    return results[0][0]


###########################################################
###########################################################


def main():
    st.title("Infinia Chatbot Application")

    # Use session_state to store the selected option
    if "option" not in st.session_state:
        st.session_state.option = ""

    # Sidebar buttons to set the option
    if st.sidebar.button("Multimodal Ingestion"):
        st.session_state.option = "Multimodal Ingest"
    elif st.sidebar.button("Multimodal Query"):
        st.session_state.option = "Multimodal query"

    if st.session_state.option == "Multimodal Ingest":
        # Use session_state to persist the uploaded file
        if "uploaded_file" not in st.session_state:
            st.session_state.uploaded_file = None

        uploaded_file = st.file_uploader("Upload a PDF file to upload:", type=["pdf"])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

        if st.button("Upload"):
            if st.session_state.uploaded_file is not None:
                with open(st.session_state.uploaded_file.name, "wb") as f:
                    f.write(st.session_state.uploaded_file.getbuffer())
                response = multimodal_ingestion(
                    pdf_path=st.session_state.uploaded_file.name
                )
                logger.info(f"Multimodal Ingestion completed!.")
                st.success("PDF has been ingested successfully!")
            else:
                st.warning("Please upload a PDF file before ingesting.")

    elif st.session_state.option == "Multimodal query":
        user_question = st.text_input("Enter your question:")
        if st.button("Submit"):
            if user_question.strip():
                tracemalloc.start()
                with st.spinner("Processing..."):
                    response = multimodal_query(query=user_question)
                tracemalloc.stop()
                if response:
                    st.success("Answer:")
                    st.write("Content:")
                    st.write(response["entity"]["text"])
                    st.write("Coordinates:")
                    st.write(response["entity"]["content_metadata"]["location"])
                    logger.info(f"Answer: {response['entity']['text']}")
                else:
                    st.error("Failed to fetch the answer.")
            else:
                st.warning("Please enter a valid question.")


if __name__ == "__main__":
    main()
