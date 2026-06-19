from pypdf import PdfReader
import asyncio
import numpy as np
from datetime import datetime

from qmix_report_writer.utils.globals import ReportState, Score
from experiments.eval import report_score, length_score

def extract_and_chunk(pdf_path, chunk_size=1000):
    """
    Extracts text from a PDF and returns a list of text chunks.
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    
    # 1. Extract text page by page
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "  # Ensure space between pages

    # 2. Chunking logic (Character-based)
    chunks = []
    for i in range(0, len(full_text), chunk_size):
        chunks.append(full_text[i : i + chunk_size])
    
    return chunks

async def run_incremental_test(chunks):
    # 1. Reset states (Optional: ensure you start fresh)
    # Depending on how your classes are built, you might need:
    # ReportState.instance().reset() 
    # Score.instance().reset()

    results = []

    print(f"--- Starting Test for {len(chunks)} Chunks ---")

    for i, chunk in enumerate(chunks):
        print(f"Processing Chunk {i+1}/{len(chunks)}...")

        # 2. Update the state with the new chunk
        # Assuming .append() adds the chunk to ReportState.instance().additions
        # and updates ReportState.instance().content (the concatenated text)
        chunk_summary = f"Summary of chunk {i+1}" 
        ReportState.instance().append(chunk, chunk_summary)

        # 3. Trigger the scoring
        # This calls your LLM and updates the Score.instance()
        quality_score = await report_score()
        length_goal = length_score(25000, 8500)

        # 4. Capture detailed metrics for this step
        # We reach into the Score and State instances to grab details
        step_data = {
            "chunk_index": i + 1,
            "chunk": chunk,
            "chunk_text_preview": chunk[:50] + "...",
            "individual_chunk_score": Score.instance().micro_scores[-1],
            "running_avg_micro": np.average(Score.instance().micro_scores),
            "global_composite_score": quality_score,
            "local_audit": Score.instance().micro_notes[-1],
            "length_score": length_goal,
            "total_score": quality_score + 0.1 * length_goal
        }
        
        results.append(step_data)
        print(f"Chunk {i+1} Score: {step_data['individual_chunk_score']:.4f} | Global: {step_data["total_score"]:.4f}")

    return results

def generate_report(results, output_file="scoring_report.md"):
    """Generates a formatted Markdown document of the results."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Incremental Scoring Evaluation Report\n\n")
        
        for data in results:
            f.write(f"## --- CHUNK {data['chunk_index']} ---\n\n")
            
            # Metric Table
            f.write("### 📊 Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| **Individual Chunk Score** | {data['individual_chunk_score']:.4f} |\n")
            f.write(f"| **Running Micro Avg** | {data['running_avg_micro']:.4f} |\n")
            f.write(f"| **Global Quality Score** | {data['global_composite_score']:.4f} |\n")
            f.write(f"| **Length Score** | {data['length_score']:.4f} |\n")
            f.write(f"| **TOTAL SCORE (Weighted)** | **{data['total_score']:.4f}** |\n\n")
            
            # Auditor Reasoning
            f.write("### 🧠 Auditor Reasoning\n")
            f.write(f"> {data['local_audit']}\n\n")
            
            # Chunk Content
            f.write("### 📄 Chunk Text\n")
            f.write("```text\n")
            f.write(f"{data['chunk']}\n")
            f.write("```\n\n")
            f.write("---\n\n")

    print(f"✅ Report generated: {output_file}")

# --- Main Execution Block ---
if __name__ == "__main__":
    report_name = ""

    # From your previous code
    pdf_chunks = extract_and_chunk("tests/test_documents/" + report_name + ".pdf", chunk_size=4000)
    
    # Run the test
    test_results = asyncio.run(run_incremental_test(pdf_chunks))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tests/scoring_results/scoring_{report_name}_{timestamp}.md"

    generate_report(test_results, filename)
