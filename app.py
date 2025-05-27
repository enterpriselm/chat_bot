from typing import List, Dict, Optional, Any
from genaitor.genaitor.core import Orchestrator, Flow, ExecutionMode, Agent, Task, AgentRole, TaskResult
from genaitor.genaitor.llm import GeminiProvider, GeminiConfig
from genaitor.genaitor.presets.tasks_objects import GeneralTask

import json

from dotenv import load_dotenv

import google.generativeai as genai
import pandas as pd
import json
import tifffile
from PIL import Image
import PyPDF2
from pptx import Presentation
import cv2
import scipy
import ezdxf
import numpy as np
import h5py
from netCDF4 import Dataset
from astropy.io import fits
import avro.datafile
import avro.schema
import trimesh
import open3d as o3d

from dotenv import load_dotenv
load_dotenv(r'.env')

class AnswerQuestionTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
    You are a knowledgeable assistant.

    **Task**: {self.description}  
    **Goal**: {self.goal}

    Using the context provided, generate a complete, accurate, and concise answer to the user question. Do not rely on external knowledge — only use the context.

    ### Input
    {input_data}

    ### Output Format
    {self.output_format}
    """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "answer_question"}
            )
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class EvaluateAnswerTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
    You are an expert evaluator of answers based on given context.

    **Task**: {self.description}  
    **Goal**: {self.goal}

    Evaluate the provided answer based solely on the context and the user’s question. Determine if the answer is complete, accurate, relevant, and aligned with the context.

    ### Input
    {input_data}

    ### Output Format
    {self.output_format}
    """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "evaluate_answer"}
            )
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class SuggestImprovementTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
    You are a skilled assistant improving model-generated answers.

    **Task**: {self.description}  
    **Goal**: {self.goal}

    Given a user question, context, and an underperforming answer, suggest a significantly improved version that is more accurate, clear, and aligned with the context.

    ### Input
    {input_data}

    ### Output Format
    {self.output_format}
    """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "suggest_improvement"}
            )
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

def extract_file_content(api_keys, file_path):
    file_format = file_path.partition('.')[2]
    
    prompt = (
        'You are a highly skilled summarization agent. Your task is to reduce the following {file_format} content to a compact, yet highly informative summary. Focus on preserving all core ideas, characters, events, arguments, and insights necessary to answer detailed questions later. Use a format that balances brevity and informativeness, similar to the Pareto principle: retain the critical 20% that conveys 80% of the value.',
        '\nOutput the summary in a structured format, with sections like:',
        '- Title and Metadata (if available)',
        '- Key Concepts or Themes',
        '- Main Events or Plot Points',
        '- Important Characters or Entities',
        '- Critical Insights or Takeaways',
        '\nEnsure that no essential information is lost, even if minor details are omitted. The summary should allow deep questioning about the original content without needing to re-read the full text.',
        '\nData: {file_content}')
    
    genai.configure(api_key=api_keys[0])
    model = genai.GenerativeModel('gemini-1.5-flash')
                
    if file_format == 'jpg' or file_format == 'png':
        img = Image.open(file_path)
        response = model.generate_content(img)
        return response.candidates[0].content.parts[0].text
    if file_format == 'csv':
        df = pd.read_csv(file_path)
        file_content = df.to_string()            
    if file_format == 'xls' or file_format == 'xlsx':
        df = pd.read_excel(file_path)
        file_content = df.to_string()
    if file_format == 'json':
        with open(file_path, 'r') as file:
            data = json.load(file)
        file_content = json.dumps(data, indent=4)
    if file_format == 'tiff':
        with tifffile.TiffFile(file_path) as tif:
            metadata = tif.pages[0].tags
            file_content = str(metadata)
    if file_format == 'pdf':
        file_content=''
        pdf = PyPDF2.PdfReader(file_path)
        for page in pdf.pages:
            file_content+=page.extract_text()
    if file_format == 'ppt':
        file_content = ""
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            file_content += run.text
    if file_format == 'mp4':
        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        descriptions = []
        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            img = buffer.tobytes()
            response = model.generate_content(img)
            descriptions.append(response.text)
        file_content = "\n".join(descriptions)
    if file_format == 'mat':
        data = scipy.io.loadmat(file_path)
        file_content = str(data)
    if file_format == 'npy' or file_format == 'npz':
        array = np.load(file_path)    
        file_content = np.array2string(array)
    if file_format == 'dxf':
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        data = []
        for entity in msp:
            data.append(str(entity.dxf))
        file_content = "\n".join(data)
    if file_format == 'hdf5':
        file_content = ""
        with h5py.File(file_path, 'r') as f:
            def print_attrs(name, obj):
                nonlocal file_content
                file_content += f"Group or Dataset: {name}\n"
                for key, val in obj.attrs.items():
                    file_content += f"  Attribute: {key} = {val}\n"
                    if isinstance(obj, h5py.Dataset):
                        file_content += f"  Data: {obj[:]}\n"
            f.visititems(print_attrs)
    if file_format == 'nc':
        file_content = ""
        with Dataset(file_path, 'r') as nc:
            for var_name in nc.variables:
                var = nc.variables[var_name]
                file_content += f"Variable: {var_name}\n"
                file_content += f"  Data: {var[:]}\n"
                for attr_name in var.ncattrs():
                    file_content += f"  Attribute: {attr_name} = {var.getncattr(attr_name)}\n"
    if file_format == 'fits':
        file_content = ""
        with fits.open(file_path) as hdul:
            for hdu in hdul:
                file_content += f"Extension: {hdu.name}\n"
                file_content += f"  Header: {hdu.header}\n"
                if hdu.data is not None:
                    file_content += f"  Data: {hdu.data}\n"
    if file_format == 'parquet':
        df = pd.read_parquet(file_path)
        file_content = df.to_string()
    if file_format == 'avro':
        file_content = ""
        with open(file_path, 'rb') as fo:
            reader = avro.datafile.Reader(fo, avro.schema.ReadersWriterSchema())
            for record in reader:
                file_content += str(record) + "\n"
    if file_format == 'ply':
        file_content=''
        mesh = trimesh.load(file_path)
        file_content = f"Vertices: {mesh.vertices}\n"
        file_content += f"Faces: {mesh.faces}\n"
    if file_format == 'pcd':
        file_content=''
        pcd = o3d.io.read_point_cloud(file_path)
        points = pcd.points
        colors = pcd.colors
        file_content = f"Points: {points}\n"
        file_content += f"Colors: {colors}\n"
    prompt = prompt.format(file_format=file_format, file_content=file_content)
    return model.generate_content(prompt)

def create_generator_agent(provider):
    generator_task = GeneralTask(
        description="Generate a concise answer based on retrieved documents and a question.",
        goal="Provide an accurate and complete answer using only the provided context.",
        output_format="A concise and precise answer to the question.",
        llm_provider=provider
    )

    return Agent(
        role=AgentRole.SPECIALIST,
        tasks=[generator_task],
        llm_provider=provider
    )

def create_evaluator_agent(provider):
    evaluator_task = GeneralTask(
        description="Evaluate the accuracy, relevance, and completeness of a generated answer.",
        goal="Provide a structured evaluation of the generated answer against the original question and context.",
        output_format=json.dumps({
            "accuracy": "0-10",
            "relevance": "0-10",
            "completeness": "0-10",
            "conciseness": "0-10",
            "hallucinations": "Yes/No",
            "suggested_improvements": "string"
        }, indent=2),
        llm_provider=provider
    )

    return Agent(
        role=AgentRole.EVALUATOR,
        tasks=[evaluator_task],
        llm_provider=provider
    )

def create_rag_orchestrator(generator_agent, evaluator_agent):
    return Orchestrator(
        agents={
            "generator_agent": generator_agent,
            "evaluator_agent": evaluator_agent
        },
        flows={
            "rag_evaluation_flow": Flow(
                agents=["generator_agent", "evaluator_agent"],
                context_pass=[True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

async def evaluate_until_threshold(query, context, orchestrator, threshold, max_attempts=5):
    feedback = ""
    generated_answer = ""
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt} for question: '{query}' ---")

        input_data = {
            "query": query,
            "relevant_docs": [context],
            "feedback": feedback,
            "retrieved_context": [context],
            "generated_answer": generated_answer
        }

        result = await orchestrator.process_request(input_data, flow_name='rag_evaluation_flow')

        if not result["success"]:
            print(f"❌ Error: {result['error']}")
            return None

        generator_output = result["content"]["generator_agent"].content.strip()
        evaluator_output_raw = result["content"]["evaluator_agent"].content.strip()

        try:
            evaluation = json.loads(evaluator_output_raw)
        except Exception:
            print("⚠️ Failed to parse evaluator output as JSON.")
            return None

        print("📌 Generator Output:\n", generator_output)
        print("📊 Evaluator Output:\n", json.dumps(evaluation, indent=2))

        # Check threshold
        if (
            int(evaluation["accuracy"]) >= threshold["accuracy"] and
            int(evaluation["relevance"]) >= threshold["relevance"] and
            int(evaluation["completeness"]) >= threshold["completeness"] and
            int(evaluation["conciseness"]) >= threshold["conciseness"] and
            evaluation["hallucinations"].strip().lower() == threshold["hallucinations"].lower()
        ):
            print("✅ Threshold met. Final answer accepted.\n")
            return generator_output

        feedback = evaluation.get("suggested_improvements", "")
        generated_answer = generator_output

    print("⚠️ Max attempts reached. Best available answer returned.\n")
    return generated_answer
