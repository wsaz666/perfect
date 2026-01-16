import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import faiss
import numpy as np
from llama_index.core.node_parser import SentenceSplitter
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import tempfile
import shutil
from typing import Optional
from openai import OpenAI
import gradio as gr
import fitz  # PyMuPDF
import chardet  # 用于自动检测编码
import hashlib
import traceback
from config import Config  # 导入配置文件
from text2vec import TextVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

VERILOG_SYSTEM_PROMPT = """
你是一位资深的 FPGA/ASIC 工程师，精通 Verilog 和 SystemVerilog。
你的任务是根据用户需求和提供的参考代码（Context），编写高质量、可综合的 Verilog 代码。

要求：
1. **代码优先**：直接输出代码，除非用户要求解释。
2. **规范性**：必须包含 module 定义、输入输出端口、parameter 定义。
3. **时序逻辑**：处理复位逻辑（通常是异步低电平复位或同步高电平复位）。
4. **风格一致**：参考 Context 中的命名规范（如 _i, _o, _r）和缩进风格。
5. **注释**：关键逻辑需要添加简短注释。
"""

# 创建知识库根目录和临时文件目录
KB_BASE_DIR = Config.kb_base_dir
os.makedirs(KB_BASE_DIR, exist_ok=True)

# 创建默认知识库目录
DEFAULT_KB = Config.default_kb
DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
os.makedirs(DEFAULT_KB_DIR, exist_ok=True)

# 创建临时输出目录
OUTPUT_DIR = Config.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("👉 正在加载本地 Embedding 模型，请稍候...", flush=True)
global_vector_model = TextVector(Config())
print("✅ 本地模型加载完成！", flush=True)

client = OpenAI(
    api_key=Config.llm_api_key,
    base_url=Config.llm_base_url
)


class DeepSeekClient:
    def generate_answer(self, system_prompt, user_prompt, model=Config.llm_model):
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()


# 获取知识库列表
def get_knowledge_bases() -> List[str]:
    """获取所有知识库名称"""
    try:
        if not os.path.exists(KB_BASE_DIR):
            os.makedirs(KB_BASE_DIR, exist_ok=True)

        kb_dirs = [d for d in os.listdir(KB_BASE_DIR)
                   if os.path.isdir(os.path.join(KB_BASE_DIR, d))]

        # 确保默认知识库存在
        if DEFAULT_KB not in kb_dirs:
            os.makedirs(os.path.join(KB_BASE_DIR, DEFAULT_KB), exist_ok=True)
            kb_dirs.append(DEFAULT_KB)

        return sorted(kb_dirs)
    except Exception as e:
        print(f"获取知识库列表失败: {str(e)}")
        return [DEFAULT_KB]


# 创建新知识库
def create_knowledge_base(kb_name: str) -> str:
    """创建新的知识库"""
    try:
        if not kb_name or not kb_name.strip():
            return "错误：知识库名称不能为空"

        # 净化知识库名称，只允许字母、数字、下划线和中文
        kb_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', kb_name.strip())

        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 已存在"

        os.makedirs(kb_path, exist_ok=True)
        return f"知识库 '{kb_name}' 创建成功"
    except Exception as e:
        return f"创建知识库失败: {str(e)}"


# 删除知识库
def delete_knowledge_base(kb_name: str) -> str:
    """删除指定的知识库"""
    try:
        if kb_name == DEFAULT_KB:
            return f"无法删除默认知识库 '{DEFAULT_KB}'"

        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 不存在"

        shutil.rmtree(kb_path)
        return f"知识库 '{kb_name}' 已删除"
    except Exception as e:
        return f"删除知识库失败: {str(e)}"


# 获取知识库文件列表
def get_kb_files(kb_name: str) -> List[str]:
    """获取指定知识库中的文件列表"""
    try:
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return []

        # 获取所有文件（排除索引文件和元数据文件）
        files = [f for f in os.listdir(kb_path)
                 if os.path.isfile(os.path.join(kb_path, f)) and
                 not f.endswith(('.index', '.json'))]

        return sorted(files)
    except Exception as e:
        print(f"获取知识库文件列表失败: {str(e)}")
        return []


# 语义分块函数
def semantic_chunk(text: str, file_ext: str, chunk_size=2000, chunk_overlap=20) -> List[str]:
    """
    语义分块：只负责切分，返回字符串列表。
    ID生成逻辑下沉到 process_and_index_files 中以保证全局唯一性。
    """
    # 1. PDF/通用 切分器
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "。", "！", "？", "；", "，", "\n", " ", ""],
        keep_separator=True
    )
    # 2. TXT 切分器 (特殊标记)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["<|endoftext|>","\n\n", "\n", ";", ""],
        keep_separator=True
    )
    # 3. 代码 切分器 (Verilog)
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["endmodule", "\n\n", "\n", ";", ""],
        keep_separator=True
    )

    file_ext = file_ext.lower()
    chunks = []

    if file_ext == ".pdf":
        chunks = pdf_splitter.split_text(text)
    elif file_ext == ".txt":
        chunks = text_splitter.split_text(text)
    elif file_ext in [".v", ".sv"]:
        chunks = code_splitter.split_text(text)
    else:
        chunks = pdf_splitter.split_text(text)

    # 过滤掉过短的块
    return [c for c in chunks if len(c) >= 20]


# 构建Faiss索引
def build_faiss_index(vector_file, index_path, metadata_path):
    try:
        with open(vector_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            raise ValueError("向量数据为空，请检查输入文件。")

        # 确认所有数据项都有向量
        valid_data = []
        for item in data:
            if 'vector' in item and item['vector']:
                valid_data.append(item)
            else:
                print(f"警告: 跳过没有向量的数据项 ID: {item.get('id', '未知')}")

        if not valid_data:
            raise ValueError("没有找到任何有效的向量数据。")

        # 提取向量
        vectors = [item['vector'] for item in valid_data]
        vectors = np.array(vectors, dtype=np.float32)

        if vectors.size == 0:
            raise ValueError("向量数组为空，转换失败。")

        # 检查向量维度
        dim = vectors.shape[1]
        n_vectors = vectors.shape[0]
        print(f"构建索引: {n_vectors} 个向量，每个向量维度: {dim}")

        # 确定索引类型和参数
        max_nlist = n_vectors // 39
        nlist = min(max_nlist, 128) if max_nlist >= 1 else 1

        if nlist >= 1 and n_vectors >= nlist * 39:
            print(f"使用 IndexIVFFlat 索引，nlist={nlist}")
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            if not index.is_trained:
                index.train(vectors)
            index.add(vectors)
        else:
            print(f"使用 IndexFlatIP 索引")
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)

        faiss.write_index(index, index_path)
        print(f"成功写入索引到 {index_path}")

        # 创建元数据
        metadata = [{'id': item['id'], 'chunk': item['chunk'], 'method': item['method']} for item in valid_data]
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"成功写入元数据到 {metadata_path}")

        return True
    except Exception as e:
        print(f"构建索引失败: {str(e)}")
        traceback.print_exc()
        raise


# 向量化文件内容
def vectorize_file(data_list, output_file_path, field_name="chunk"):
    """向量化文件内容，处理长度限制并确保输入有效"""
    if not data_list:
        print("警告: 没有数据需要向量化")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return

    # 准备查询文本，确保每个文本有效且长度适中
    valid_data = []
    valid_texts = []

    for data in data_list:
        text = data.get(field_name, "")
        # 确保文本不为空且长度合适
        if text and 1 <= len(text) <= 8000:  # 略小于API限制的8192，留出一些余量
            valid_data.append(data)
            valid_texts.append(text)
        else:
            # 如果文本太长，截断它
            if len(text) > 8000:
                truncated_text = text[:8000]
                print(f"警告: 文本过长，已截断至8000字符。原始长度: {len(text)}")
                data[field_name] = truncated_text
                valid_data.append(data)
                valid_texts.append(truncated_text)
            else:
                print(f"警告: 跳过空文本或长度为0的文本")

    if not valid_texts:
        print("错误: 所有文本都无效，无法进行向量化")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return

    # 向量化有效文本
    vectors = vectorize_query(valid_texts)

    # 检查向量化是否成功
    if vectors.size == 0 or len(vectors) != len(valid_data):
        print(
            f"错误: 向量化失败或向量数量({len(vectors) if vectors.size > 0 else 0})与数据条目({len(valid_data)})不匹配")
        # 保存原始数据，但不含向量
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(valid_data, outfile, ensure_ascii=False, indent=4)
        return

    # 添加向量到数据中
    for data, vector in zip(valid_data, vectors):
        data['vector'] = vector.tolist()

    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(valid_data, outfile, ensure_ascii=False, indent=4)

    print(f"成功向量化 {len(valid_data)} 条数据并保存到 {output_file_path}")


# 向量化查询 - 通用函数，被多处使用
def vectorize_query(query, model_name=Config.model_name, batch_size=Config.batch_size) -> np.ndarray:
    """
    向量化文本查询（已修改为调用 text2vec 本地模型）
    """
    if not query:
        return np.array([])

    if isinstance(query, str):
        query = [query]

    # 过滤无效文本
    valid_queries = [q for q in query if q and isinstance(q, str) and q.strip()]
    if not valid_queries:
        return np.array([])

    try:
        # 👇 直接调用全局的本地模型实例
        # get_vec_batch 会自动处理 batch，并且因为是本地模型，没有 TPM 限制
        print(f"🚀 [本地] 正在向量化 {len(valid_queries)} 条文本...", flush=True)

        # text2vec 返回的是 Torch Tensor，需要转成 Numpy 给 Faiss 用
        vectors = global_vector_model.get_vec_batch(valid_queries, bs=batch_size)

        if isinstance(vectors, torch.Tensor):
            return vectors.cpu().numpy()
        elif isinstance(vectors, list):
            return np.array(vectors)
        else:
            return np.array(vectors)

    except Exception as e:
        print(f"❌ 向量化失败: {str(e)}")
        traceback.print_exc()
        return np.array([])


# 简单的向量搜索，用于基本对比
def vector_search(query, index_path, metadata_path, limit):
    """基本向量搜索函数"""
    query_vector = vectorize_query(query)
    if query_vector.size == 0:
        return []

    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    index = faiss.read_index(index_path)
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except UnicodeDecodeError:
        print(f"警告：{metadata_path} 包含非法字符，使用 UTF-8 忽略错误重新加载")
        with open(metadata_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            metadata = json.loads(content)

    D, I = index.search(query_vector, limit)
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return results


def clean_text(text):
    """清理文本中的非法字符，控制文本长度"""
    if not text:
        return ""
    # 移除控制字符，保留换行和制表符
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # 移除重复的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 确保文本长度在合理范围内
    return text.strip()


# PDF文本提取
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            # 清理不可打印字符，尝试用 UTF-8 解码，失败时忽略非法字符
            text += page_text.encode('utf-8', errors='ignore').decode('utf-8')
        if not text.strip():
            print(f"警告：PDF文件 {pdf_path} 提取内容为空")
        return text
    except Exception as e:
        print(f"PDF文本提取失败：{str(e)}")
        return ""


# 处理单个文件
def process_single_file(file_path: str) -> str:
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            if not text:
                return f"PDF文件 {file_path} 内容为空或无法提取"
        else:
            with open(file_path, "rb") as f:
                content = f.read()
            result = chardet.detect(content)
            detected_encoding = result['encoding']
            confidence = result['confidence']

            # 尝试多种编码方式
            if detected_encoding and confidence > 0.7:
                try:
                    text = content.decode(detected_encoding)
                    print(f"文件 {file_path} 使用检测到的编码 {detected_encoding} 解码成功")
                except UnicodeDecodeError:
                    text = content.decode('utf-8', errors='ignore')
                    print(f"文件 {file_path} 使用 {detected_encoding} 解码失败，强制使用 UTF-8 忽略非法字符")
            else:
                # 尝试多种常见编码
                encodings = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin-1', 'utf-16', 'cp936', 'big5']
                text = None
                for encoding in encodings:
                    try:
                        text = content.decode(encoding)
                        print(f"文件 {file_path} 使用 {encoding} 解码成功")
                        break
                    except UnicodeDecodeError:
                        continue

                # 如果所有编码都失败，使用忽略错误的方式解码
                if text is None:
                    text = content.decode('utf-8', errors='ignore')
                    print(f"警告：文件 {file_path} 使用 UTF-8 忽略非法字符")

        # 确保文本是干净的，移除非法字符
        text = clean_text(text)
        return text
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        traceback.print_exc()
        return f"处理文件 {file_path} 失败：{str(e)}"


# 批量处理并索引文件 - 修改为支持指定知识库
def process_and_index_files(file_objs: List, kb_name: str = "default_kb") -> str:
    print(f"！！！进入文件处理函数 - 目标知识库: {kb_name}！！！", flush=True)

    # 1. 路径准备
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    os.makedirs(kb_dir, exist_ok=True)

    # 临时文件夹仅用于暂存上传的原始文件流，用完即删
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 🟢 核心修正：中间数据文件必须存在各自的知识库目录下，实现隔离！
    # 这样 KB_A 的数据永远不会跑到 KB_B 里去
    semantic_chunk_output = os.path.join(kb_dir, "dataset_source.json")  # 纯文本账本
    semantic_chunk_vector = os.path.join(kb_dir, "dataset_vector.json")  # 向量账本

    # 最终索引文件
    semantic_chunk_index = os.path.join(kb_dir, "semantic_chunk.index")
    semantic_chunk_metadata = os.path.join(kb_dir, "semantic_chunk_metadata.json")

    new_chunks_buffer = []
    error_messages = []

    try:
        if not file_objs: return "错误：没有选择任何文件"
        print(f"开始处理 {len(file_objs)} 个文件...")

        for file_obj in file_objs:
            # 获取路径和文件名
            source_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
            filename = os.path.basename(source_path)
            file_ext = os.path.splitext(filename)[1].lower()

            # 构造临时路径 (避免 Windows 锁)
            tmp_filename = f"proc_{hashlib.md5(filename.encode()).hexdigest()}{file_ext}"
            tmp_path = os.path.join(OUTPUT_DIR, tmp_filename)

            try:
                # 复制文件到临时目录
                shutil.copy2(source_path, tmp_path)
                # 备份原始文件到知识库目录 (留档)
                shutil.copy2(tmp_path, os.path.join(kb_dir, filename))

                # 读取文本
                raw_text = ""
                if file_ext == ".pdf":
                    try:
                        doc = fitz.open(tmp_path)
                        text_list = [page.get_text() for page in doc]
                        raw_text = "\n\n".join(text_list)
                        doc.close()
                    except Exception as e:
                        print(f"PDF解析失败: {e}")
                        # 备用方案：LangChain Loader
                        try:
                            loader = PyMuPDFLoader(tmp_path)
                            pages = loader.load()
                            raw_text = "\n\n".join([p.page_content for p in pages])
                        except:
                            error_messages.append(f"{filename} 解析失败")
                            continue
                else:
                    # 文本/代码文件读取
                    with open(tmp_path, "rb") as f:
                        content_bytes = f.read()

                    decoded = False
                    for enc in ['utf-8', 'gbk', 'gb18030', 'latin-1']:
                        try:
                            raw_text = content_bytes.decode(enc)
                            decoded = True
                            break
                        except:
                            continue
                    if not decoded:
                        raw_text = content_bytes.decode('utf-8', errors='ignore')

                # 分块
                raw_text = clean_text(raw_text)
                if raw_text and len(raw_text) > 10:
                    chunks_str_list = semantic_chunk(raw_text, file_ext)

                    for chunk_text in chunks_str_list:
                        # 生成唯一ID
                        unique_str = f"{filename}_{chunk_text}"
                        chunk_id = hashlib.md5(unique_str.encode('utf-8')).hexdigest()

                        new_chunks_buffer.append({
                            "id": chunk_id,
                            "chunk": chunk_text,
                            "metadata": {"source": filename},
                            "method": "semantic_chunk"
                        })
                else:
                    error_messages.append(f"文件 {filename} 内容为空")

            except Exception as e:
                error_messages.append(f"文件 {filename} 处理异常: {str(e)}")
                traceback.print_exc()
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

        if not new_chunks_buffer:
            return "没有生成任何有效分块。\n" + "\n".join(error_messages)

        # 3. 增量更新 (读取当前知识库专属的 JSON)
        final_all_chunks = []
        if os.path.exists(semantic_chunk_output):
            try:
                with open(semantic_chunk_output, 'r', encoding='utf-8') as f:
                    final_all_chunks = json.load(f)
            except:
                final_all_chunks = []

        # 合并去重
        chunk_map = {item["id"]: item for item in final_all_chunks}
        for item in new_chunks_buffer:
            chunk_map[item["id"]] = item
        final_all_chunks = list(chunk_map.values())

        # 写回当前知识库目录
        with open(semantic_chunk_output, 'w', encoding='utf-8') as f:
            json.dump(final_all_chunks, f, ensure_ascii=False, indent=4)

        print(f"数据合并完成。当前知识库 {kb_name} 总条目: {len(final_all_chunks)}")

        # 4. 向量化
        print("开始全量向量化...")
        vectorize_file(final_all_chunks, semantic_chunk_vector)

        # 5. 构建索引
        print("开始构建索引...")
        build_faiss_index(semantic_chunk_vector, semantic_chunk_index, semantic_chunk_metadata)

        status_msg = f"成功！知识库 '{kb_name}' 更新完毕。\n总条目: {len(final_all_chunks)}\n本次新增: {len(new_chunks_buffer)}"
        if error_messages:
            status_msg += "\n\n⚠️ 部分文件警告:\n" + "\n".join(error_messages)
        return status_msg

    except Exception as e:
        traceback.print_exc()
        return f"严重错误: {str(e)}"


# 核心联网搜索功能
def get_search_background(query: str, max_length: int = 1500) -> str:
    try:
        from retrievor import q_searching
        search_results = q_searching(query)
        cleaned_results = re.sub(r'\s+', ' ', search_results).strip()
        return cleaned_results[:max_length]
    except Exception as e:
        print(f"联网搜索失败：{str(e)}")
        return ""


# 基本的回答生成
def generate_answer_from_deepseek(question: str, system_prompt: str = "你是一位资深的 FPGA/ASIC 工程师，精通 Verilog 和 SystemVerilog，你的任务是根据背景知识，编写高质量、可综合的 Verilog 代码。",
                                  background_info: Optional[str] = None) -> str:
    deepseek_client = DeepSeekClient()
    user_prompt = f"问题：{question}"
    if background_info:
        user_prompt = f"背景知识：{background_info}\n\n{user_prompt}"
    try:
        answer = deepseek_client.generate_answer(system_prompt, user_prompt)
        return answer
    except Exception as e:
        return f"生成回答时出错：{str(e)}"


# 多跳推理RAG系统 - 核心创新点
class ReasoningRAG:
    """
    多跳推理RAG系统，通过迭代式的检索和推理过程回答问题，支持流式响应
    """

    def __init__(self,
                 index_path: str,
                 metadata_path: str,
                 max_hops: int = 3,
                 initial_candidates: int = 5,
                 refined_candidates: int = 3,
                 reasoning_model: str = Config.llm_model,
                 verbose: bool = False):
        """
        初始化推理RAG系统

        参数:
            index_path: FAISS索引的路径
            metadata_path: 元数据JSON文件的路径
            max_hops: 最大推理-检索跳数
            initial_candidates: 初始检索候选数量
            refined_candidates: 精炼检索候选数量
            reasoning_model: 用于推理步骤的LLM模型
            verbose: 是否打印详细日志
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.max_hops = max_hops
        self.initial_candidates = initial_candidates
        self.refined_candidates = refined_candidates
        self.reasoning_model = reasoning_model
        self.verbose = verbose

        # 加载索引和元数据
        self._load_resources()

    def _load_resources(self):
        """加载FAISS索引和元数据"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except UnicodeDecodeError:
                with open(self.metadata_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    self.metadata = json.loads(content)
        else:
            raise FileNotFoundError(f"Index or metadata not found at {self.index_path} or {self.metadata_path}")

    def _vectorize_query(self, query: str) -> np.ndarray:
        """将查询转换为向量"""
        return vectorize_query(query).reshape(1, -1)

    def _retrieve(self, query_vector: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """使用向量相似性检索块"""
        if query_vector.size == 0:
            return []

        D, I = self.index.search(query_vector, limit)
        results = [self.metadata[i] for i in I[0] if i < len(self.metadata)]
        return results

    def _generate_reasoning(self,
                            query: str,
                            retrieved_chunks: List[Dict[str, Any]],
                            previous_queries: List[str] = None,
                            hop_number: int = 0) -> Dict[str, Any]:
        """
        为检索到的信息生成推理分析并识别信息缺口

        返回包含以下字段的字典:
            - analysis: 对当前信息的推理分析
            - missing_info: 已识别的缺失信息
            - follow_up_queries: 填补信息缺口的后续查询列表
            - is_sufficient: 表示信息是否足够的布尔值
        """
        if previous_queries is None:
            previous_queries = []

        # 为模型准备上下文
        chunks_text = "\n\n".join([f"[Chunk {i + 1}]: {chunk['chunk']}"
                                   for i, chunk in enumerate(retrieved_chunks)])

        previous_queries_text = "\n".join([f"Q{i + 1}: {q}" for i, q in enumerate(previous_queries)])

        system_prompt = """
        你是 Verilog 代码库的专家分析系统。
        你的任务是分析检索到的代码片段，识别逻辑缺口（如缺失的子模块、未定义的参数），并提出后续查询以补全代码上下文。
        """

        user_prompt = f"""
        ## 原始查询
        {query}

        ## 先前查询（如果有）
        {previous_queries_text if previous_queries else "无"}

        ## 检索到的信息（跳数 {hop_number}）
        {chunks_text if chunks_text else "未检索到信息。"}

        ## 你的任务
        1. 分析已检索到的信息与原始查询的关系
        2. 确定能够更完整回答查询的特定缺失信息
        3. 提出1-3个针对性的后续查询，以检索缺失信息
        4. 确定当前信息是否足够回答原始查询

        以JSON格式回答，包含以下字段:
        - analysis: 对当前信息的详细分析
        - missing_info: 特定缺失信息的列表
        - follow_up_queries: 1-3个具体的后续查询
        - is_sufficient: 表示信息是否足够的布尔值
        
        ⚠️ **重要逻辑规则（必须严格遵守）**：
        1. **如果你在 "missing_info" 字段中列出了任何内容，那么 "is_sufficient" 必须为 false！**
        2. 只有当你完全确定当前检索到的 Chunk 已经包含了用户问题所需的所有细节（无需任何额外查询）时，"is_sufficient" 才能为 true。
        3. 如果代码被截断（Chunking）导致关键部分（如行号、中间逻辑）丢失，必须标记为不完整。
        """

        try:
            response = client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            reasoning_text = response.choices[0].message.content.strip()

            # 解析JSON响应
            try:
                reasoning = json.loads(reasoning_text)
                # 确保预期的键存在
                required_keys = ["analysis", "missing_info", "follow_up_queries", "is_sufficient"]
                for key in required_keys:
                    if key not in reasoning:
                        reasoning[key] = [] if key != "is_sufficient" else False
                return reasoning
            except json.JSONDecodeError:
                # 如果JSON解析失败，则回退
                if self.verbose:
                    print(f"无法从模型输出解析JSON: {reasoning_text[:100]}...")
                return {
                    "analysis": "无法分析检索到的信息。",
                    "missing_info": ["无法识别缺失信息"],
                    "follow_up_queries": [],
                    "is_sufficient": False
                }

        except Exception as e:
            if self.verbose:
                print(f"推理生成错误: {e}")
                print(traceback.format_exc())
            return {
                "analysis": "分析过程出错。",
                "missing_info": [],
                "follow_up_queries": [],
                "is_sufficient": False
            }

    def _synthesize_answer(self,
                           query: str,
                           all_chunks: List[Dict[str, Any]],
                           reasoning_steps: List[Dict[str, Any]],
                           use_table_format: bool = False) -> str:
        """从所有检索到的块和推理步骤中合成最终答案"""
        # 合并所有块，去除重复
        unique_chunks = []
        chunk_ids = set()
        for chunk in all_chunks:
            if chunk["id"] not in chunk_ids:
                unique_chunks.append(chunk)
                chunk_ids.add(chunk["id"])

        # 准备上下文
        chunks_text = "\n\n".join([f"[Chunk {i + 1}]: {chunk['chunk']}"
                                   for i, chunk in enumerate(unique_chunks)])

        # 准备推理跟踪
        reasoning_trace = ""
        for i, step in enumerate(reasoning_steps):
            reasoning_trace += f"\n\n推理步骤 {i + 1}:\n"
            reasoning_trace += f"分析: {step['analysis']}\n"
            reasoning_trace += f"缺失信息: {', '.join(step['missing_info'])}\n"
            reasoning_trace += f"后续查询: {', '.join(step['follow_up_queries'])}"

        system_prompt = """
        你是一位资深的 FPGA/ASIC 工程师，精通 Verilog 和 SystemVerilog。
        你的任务是根据用户需求和提供的参考代码（Context），编写高质量、可综合的 Verilog 代码。

        要求：
        1. **代码优先**：直接输出代码，除非用户要求解释。
        2. **规范性**：必须包含 module 定义、输入输出端口、parameter 定义。
        3. **时序逻辑**：处理复位逻辑（通常是异步低电平复位或同步高电平复位）。
        4. **风格一致**：参考 Context 中的命名规范（如 _i, _o, _r）和缩进风格。
        5. **注释**：关键逻辑需要添加简短注释。
        """

        output_format_instruction = ""
        if use_table_format:
            output_format_instruction = """
            请尽可能以Markdown表格格式组织你的回答。如果信息适合表格形式展示，请使用表格；
            如果不适合表格形式，可以先用文本介绍，然后再使用表格总结关键信息。

            表格语法示例：
            | 标题1 | 标题2 | 标题3 |
            | ----- | ----- | ----- |
            | 内容1 | 内容2 | 内容3 |

            确保表格格式符合Markdown标准，以便正确渲染。
            """

        user_prompt = f"""
        ## 原始查询
        {query}

        ## 检索到的信息块
        {chunks_text}

        ## 推理过程
        {reasoning_trace}

        ## 你的任务
        使用提供的信息块为原始查询合成一个全面的答案。你的答案应该:

        1. 直接回应查询
        2. 结构清晰，易于理解
        3. 基于检索到的信息
        4. 承认可用信息中的任何重大缺口

        {output_format_instruction}

        以直接回应提出原始查询的用户的方式呈现你的答案。
        """

        try:
            response = client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"答案合成错误: {e}")
                print(traceback.format_exc())
            return "由于出错，无法生成答案。"

    def stream_retrieve_and_answer(self, query: str, use_table_format: bool = False):
        """
        执行多跳检索和回答生成的流式方法，逐步返回结果

        这是一个生成器函数，会在处理的每个阶段产生中间结果
        """
        all_chunks = []
        all_queries = [query]
        reasoning_steps = []

        # 生成状态更新
        yield {
            "status": "正在将查询向量化...",
            "reasoning_display": "",
            "answer": None,
            "all_chunks": [],
            "reasoning_steps": []
        }

        # 初始检索
        try:
            query_vector = self._vectorize_query(query)
            if query_vector.size == 0:
                yield {
                    "status": "向量化失败",
                    "reasoning_display": "由于嵌入错误，无法处理查询。",
                    "answer": "由于嵌入错误，无法处理查询。",
                    "all_chunks": [],
                    "reasoning_steps": []
                }
                return

            yield {
                "status": "正在执行初始检索...",
                "reasoning_display": "",
                "answer": None,
                "all_chunks": [],
                "reasoning_steps": []
            }

            initial_chunks = self._retrieve(query_vector, self.initial_candidates)
            all_chunks.extend(initial_chunks)

            if not initial_chunks:
                yield {
                    "status": "未找到相关信息",
                    "reasoning_display": "未找到与您的查询相关的信息。",
                    "answer": "未找到与您的查询相关的信息。",
                    "all_chunks": [],
                    "reasoning_steps": []
                }
                return

            # 更新状态，展示找到的初始块
            chunks_preview = "\n".join([f"- {chunk['chunk'][:100]}..." for chunk in initial_chunks[:2]])
            yield {
                "status": f"找到 {len(initial_chunks)} 个相关信息块，正在生成初步分析...",
                "reasoning_display": f"### 检索到的初始信息\n{chunks_preview}\n\n### 正在分析...",
                "answer": None,
                "all_chunks": all_chunks,
                "reasoning_steps": []
            }

            # 初始推理
            reasoning = self._generate_reasoning(query, initial_chunks, hop_number=0)
            reasoning_steps.append(reasoning)

            # 生成当前的推理显示
            reasoning_display = "### 多跳推理过程\n"
            reasoning_display += f"**推理步骤 1**\n"
            reasoning_display += f"- 分析: {reasoning['analysis'][:200]}...\n"
            reasoning_display += f"- 缺失信息: {', '.join(reasoning['missing_info'])}\n"
            if reasoning['follow_up_queries']:
                reasoning_display += f"- 后续查询: {', '.join(reasoning['follow_up_queries'])}\n"
            reasoning_display += f"- 信息是否足够: {'是' if reasoning['is_sufficient'] else '否'}\n\n"

            yield {
                "status": "初步分析完成",
                "reasoning_display": reasoning_display,
                "answer": None,
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }

            # 检查是否需要额外的跳数
            hop = 1
            while (hop < self.max_hops and
                   not reasoning["is_sufficient"] and
                   reasoning["follow_up_queries"]):

                follow_up_status = f"执行跳数 {hop}，正在处理 {len(reasoning['follow_up_queries'])} 个后续查询..."
                yield {
                    "status": follow_up_status,
                    "reasoning_display": reasoning_display + f"\n\n### {follow_up_status}",
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }

                hop_chunks = []

                # 处理每个后续查询
                for i, follow_up_query in enumerate(reasoning["follow_up_queries"]):
                    all_queries.append(follow_up_query)

                    query_status = f"处理后续查询 {i + 1}/{len(reasoning['follow_up_queries'])}: {follow_up_query}"
                    yield {
                        "status": query_status,
                        "reasoning_display": reasoning_display + f"\n\n### {query_status}",
                        "answer": None,
                        "all_chunks": all_chunks,
                        "reasoning_steps": reasoning_steps
                    }

                    # 为后续查询检索
                    follow_up_vector = self._vectorize_query(follow_up_query)
                    if follow_up_vector.size > 0:
                        follow_up_chunks = self._retrieve(follow_up_vector, self.refined_candidates)
                        hop_chunks.extend(follow_up_chunks)
                        all_chunks.extend(follow_up_chunks)

                        # 更新状态，显示新找到的块数量
                        yield {
                            "status": f"查询 '{follow_up_query}' 找到了 {len(follow_up_chunks)} 个相关块",
                            "reasoning_display": reasoning_display + f"\n\n为查询 '{follow_up_query}' 找到了 {len(follow_up_chunks)} 个相关块",
                            "answer": None,
                            "all_chunks": all_chunks,
                            "reasoning_steps": reasoning_steps
                        }

                # 为此跳数生成推理
                yield {
                    "status": f"正在为跳数 {hop} 生成推理分析...",
                    "reasoning_display": reasoning_display + f"\n\n### 正在为跳数 {hop} 生成推理分析...",
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }

                reasoning = self._generate_reasoning(
                    query,
                    hop_chunks,
                    previous_queries=all_queries[:-1],
                    hop_number=hop
                )
                reasoning_steps.append(reasoning)

                # 更新推理显示
                reasoning_display += f"\n**推理步骤 {hop + 1}**\n"
                reasoning_display += f"- 分析: {reasoning['analysis'][:200]}...\n"
                reasoning_display += f"- 缺失信息: {', '.join(reasoning['missing_info'])}\n"
                if reasoning['follow_up_queries']:
                    reasoning_display += f"- 后续查询: {', '.join(reasoning['follow_up_queries'])}\n"
                reasoning_display += f"- 信息是否足够: {'是' if reasoning['is_sufficient'] else '否'}\n"

                yield {
                    "status": f"跳数 {hop} 完成",
                    "reasoning_display": reasoning_display,
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }

                hop += 1

            # 合成最终答案
            yield {
                "status": "正在合成最终答案...",
                "reasoning_display": reasoning_display + "\n\n### 正在合成最终答案...",
                "answer": "正在处理您的问题，请稍候...",
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }

            answer = self._synthesize_answer(query, all_chunks, reasoning_steps, use_table_format)

            # 为最终显示准备检索内容汇总
            all_chunks_summary = "\n\n".join([f"**检索块 {i + 1}**:\n{chunk['chunk']}"
                                              for i, chunk in enumerate(all_chunks[:10])])  # 限制显示前10个块

            if len(all_chunks) > 10:
                all_chunks_summary += f"\n\n...以及另外 {len(all_chunks) - 10} 个块（总计 {len(all_chunks)} 个）"

            enhanced_display = reasoning_display + "\n\n### 检索到的内容\n" + all_chunks_summary + "\n\n### 回答已生成"

            yield {
                "status": "回答已生成",
                "reasoning_display": enhanced_display,
                "answer": answer,
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }

        except Exception as e:
            error_msg = f"处理过程中出错: {str(e)}"
            if self.verbose:
                print(error_msg)
                print(traceback.format_exc())

            yield {
                "status": "处理出错",
                "reasoning_display": error_msg,
                "answer": f"处理您的问题时遇到错误: {str(e)}",
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }

    def retrieve_and_answer(self, query: str, use_table_format: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        执行多跳检索和回答生成的主要方法

        返回:
            包含以下内容的元组:
            - 最终答案
            - 包含推理步骤和所有检索到的块的调试字典
        """
        all_chunks = []
        all_queries = [query]
        reasoning_steps = []
        debug_info = {"reasoning_steps": [], "all_chunks": [], "all_queries": all_queries}

        # 初始检索
        query_vector = self._vectorize_query(query)
        if query_vector.size == 0:
            return "由于嵌入错误，无法处理查询。", debug_info

        initial_chunks = self._retrieve(query_vector, self.initial_candidates)
        all_chunks.extend(initial_chunks)
        debug_info["all_chunks"].extend(initial_chunks)

        if not initial_chunks:
            return "未找到与您的查询相关的信息。", debug_info

        # 初始推理
        reasoning = self._generate_reasoning(query, initial_chunks, hop_number=0)
        reasoning_steps.append(reasoning)
        debug_info["reasoning_steps"].append(reasoning)

        # 检查是否需要额外的跳数
        hop = 1
        while (hop < self.max_hops and
               not reasoning["is_sufficient"] and
               reasoning["follow_up_queries"]):

            if self.verbose:
                print(f"开始跳数 {hop}，有 {len(reasoning['follow_up_queries'])} 个后续查询")

            hop_chunks = []

            # 处理每个后续查询
            for follow_up_query in reasoning["follow_up_queries"]:
                all_queries.append(follow_up_query)
                debug_info["all_queries"].append(follow_up_query)

                # 为后续查询检索
                follow_up_vector = self._vectorize_query(follow_up_query)
                if follow_up_vector.size > 0:
                    follow_up_chunks = self._retrieve(follow_up_vector, self.refined_candidates)
                    hop_chunks.extend(follow_up_chunks)
                    all_chunks.extend(follow_up_chunks)
                    debug_info["all_chunks"].extend(follow_up_chunks)

            # 为此跳数生成推理
            reasoning = self._generate_reasoning(
                query,
                hop_chunks,
                previous_queries=all_queries[:-1],
                hop_number=hop
            )
            reasoning_steps.append(reasoning)
            debug_info["reasoning_steps"].append(reasoning)

            hop += 1

        # 合成最终答案
        answer = self._synthesize_answer(query, all_chunks, reasoning_steps, use_table_format)

        return answer, debug_info


# 基于选定知识库生成索引路径
def get_kb_paths(kb_name: str) -> Dict[str, str]:
    """获取指定知识库的索引文件路径"""
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    return {
        "index_path": os.path.join(kb_dir, "semantic_chunk.index"),
        "metadata_path": os.path.join(kb_dir, "semantic_chunk_metadata.json")
    }


def multi_hop_generate_answer(query: str, kb_name: str, use_table_format: bool = False,
                              system_prompt: str = "你是一名verilog专家。") -> Tuple[str, Dict]:
    """使用多跳推理RAG生成答案，基于指定知识库"""
    kb_paths = get_kb_paths(kb_name)

    reasoning_rag = ReasoningRAG(
        index_path=kb_paths["index_path"],
        metadata_path=kb_paths["metadata_path"],
        max_hops=3,
        initial_candidates=5,
        refined_candidates=3,
        reasoning_model=Config.llm_model,
        verbose=True
    )

    answer, debug_info = reasoning_rag.retrieve_and_answer(query, use_table_format)
    return answer, debug_info


# 使用简单向量检索生成答案，基于指定知识库
def simple_generate_answer(query: str, kb_name: str, use_table_format: bool = False) -> str:
    """使用简单的向量检索生成答案，不使用多跳推理"""
    try:
        kb_paths = get_kb_paths(kb_name)

        # 使用基本向量搜索
        search_results = vector_search(query, kb_paths["index_path"], kb_paths["metadata_path"], limit=5)

        if not search_results:
            return "未找到相关信息。"

        # 准备背景信息
        background_chunks = "\n\n".join([
            f"// --- [Reference Code {i + 1}] ---\n{result['chunk']}\n// ---------------------------"
            for i, result in enumerate(search_results)
        ])

        # 生成答案
        system_prompt = VERILOG_SYSTEM_PROMPT

        if use_table_format:
            system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"

        user_prompt = f"""
        问题：{query}

        背景信息：
        {background_chunks}

        请基于以上背景信息回答用户的问题。
        """

        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"生成答案时出错：{str(e)}"


# 修改主要的问题处理函数以支持指定知识库
def ask_question_parallel(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True,
                          use_table_format: bool = False, multi_hop: bool = False) -> str:
    """基于指定知识库回答问题"""
    try:
        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]

        search_background = ""
        local_answer = ""
        debug_info = {}

        # 并行处理
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if use_search:
                futures[executor.submit(get_search_background, question)] = "search"

            if os.path.exists(index_path):
                if multi_hop:
                    # 使用多跳推理
                    futures[executor.submit(multi_hop_generate_answer, question, kb_name, use_table_format)] = "rag"
                else:
                    # 使用简单向量检索
                    futures[executor.submit(simple_generate_answer, question, kb_name, use_table_format)] = "simple"

            for future in as_completed(futures):
                result = future.result()
                if futures[future] == "search":
                    search_background = result or ""
                elif futures[future] == "rag":
                    local_answer, debug_info = result
                elif futures[future] == "simple":
                    local_answer = result

        # 如果同时有搜索和本地结果，合并它们
        if search_background and local_answer:
            system_prompt = "你是一名硬件工程师，请整合网络资料和本地代码库，编写 Verilog 模块。"

            table_instruction = ""
            if use_table_format:
                table_instruction = """
                请尽可能以Markdown表格的形式呈现你的回答，特别是对于症状、治疗方法、药物等结构化信息。

                请确保你的表格遵循正确的Markdown语法：
                | 列标题1 | 列标题2 | 列标题3 |
                | ------- | ------- | ------- |
                | 数据1   | 数据2   | 数据3   |
                """

            user_prompt = f"""
            问题：{question}

            网络搜索结果：{search_background}

            本地知识库分析：{local_answer}

            {table_instruction}

            请根据以上信息，提供一个综合的回答。
            """

            try:
                response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                combined_answer = response.choices[0].message.content.strip()
                return combined_answer
            except Exception as e:
                # 如果合并失败，回退到本地答案
                return local_answer
        elif local_answer:
            return local_answer
        elif search_background:
            # 仅从搜索结果生成答案
            system_prompt = VERILOG_SYSTEM_PROMPT
            if use_table_format:
                system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
            return generate_answer_from_deepseek(question, system_prompt=system_prompt,
                                                 background_info=f"[联网搜索结果]：{search_background}")
        else:
            return "未找到相关信息。"

    except Exception as e:
        return f"查询失败：{str(e)}"


# 修改以支持多知识库的流式响应函数
def process_question_with_reasoning(question: str, kb_name: str = DEFAULT_KB, use_search: bool = True,
                                    use_table_format: bool = False, multi_hop: bool = False, chat_history: List = None):
    """增强版process_question，支持流式响应，实时显示检索和推理过程，支持多知识库和对话历史"""
    try:
        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]

        # 构建带对话历史的问题
        if chat_history and len(chat_history) > 0:
            # 构建对话上下文
            context = "之前的对话内容：\n"
            for user_msg, assistant_msg in chat_history[-3:]:  # 只取最近3轮对话
                context += f"用户：{user_msg}\n"
                context += f"助手：{assistant_msg}\n"
            context += f"\n当前问题：{question}"
            enhanced_question = f"基于以下对话历史，回答用户的当前问题。\n{context}"
        else:
            enhanced_question = question

        # 初始状态
        search_result = "联网搜索进行中..." if use_search else "未启用联网搜索"

        if multi_hop:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行多跳推理检索..."
            search_display = f"### 联网搜索结果\n{search_result}\n\n### 推理状态\n{reasoning_status}"
            yield search_display, "正在启动多跳推理流程..."
        else:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行向量检索..."
            search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n{reasoning_status}"
            yield search_display, "正在启动简单检索流程..."

        # 如果启用，并行运行搜索
        search_future = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            if use_search:
                search_future = executor.submit(get_search_background, question)

        # 检查索引是否存在
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            # 如果索引不存在，提前返回
            if search_future:
                # 等待搜索结果
                search_result = "等待联网搜索结果..."
                search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n知识库 '{kb_name}' 中未找到索引"
                yield search_display, "等待联网搜索结果..."

                search_result = search_future.result() or "未找到相关网络信息"
                system_prompt = VERILOG_SYSTEM_PROMPT
                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                answer = generate_answer_from_deepseek(enhanced_question, system_prompt=system_prompt,
                                                       background_info=f"[联网搜索结果]：{search_result}")

                search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n无法在知识库 '{kb_name}' 中进行本地检索（未找到索引）"
                yield search_display, answer
            else:
                yield f"知识库 '{kb_name}' 中未找到索引，且未启用联网搜索", "无法回答您的问题。请先上传文件到该知识库或启用联网搜索。"
            return

        # 开始流式处理
        current_answer = "正在分析您的问题..."

        if multi_hop:
            # 使用多跳推理的流式接口
            reasoning_rag = ReasoningRAG(
                index_path=index_path,
                metadata_path=metadata_path,
                max_hops=3,
                initial_candidates=5,
                refined_candidates=3,
                verbose=True
            )

            # 使用enhanced_question进行检索
            for step_result in reasoning_rag.stream_retrieve_and_answer(enhanced_question, use_table_format):
                # 更新当前状态
                status = step_result["status"]
                reasoning_display = step_result["reasoning_display"]

                # 如果有新的答案，更新
                if step_result["answer"]:
                    current_answer = step_result["answer"]

                # 如果搜索结果已返回，更新搜索结果
                if search_future and search_future.done():
                    search_result = search_future.result() or "未找到相关网络信息"

                # 构建并返回当前状态
                current_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 推理状态\n{status}\n\n{reasoning_display}"
                yield current_display, current_answer
        else:
            # 简单向量检索的流式处理
            yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n正在执行向量相似度搜索...", "正在检索相关信息..."

            # 执行简单向量搜索，使用enhanced_question
            try:
                search_results = vector_search(enhanced_question, index_path, metadata_path, limit=5)

                if not search_results:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n未找到相关信息", f"知识库 '{kb_name}' 中未找到相关信息。"
                    current_answer = f"知识库 '{kb_name}' 中未找到相关信息。"
                else:
                    # 显示检索到的信息
                    chunks_detail = "\n\n".join(
                        [f"**相关信息 {i + 1}**:\n{result['chunk']}" for i, result in enumerate(search_results[:5])])
                    chunks_preview = "\n".join(
                        [f"- {result['chunk'][:100]}..." for i, result in enumerate(search_results[:3])])
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n找到 {len(search_results)} 个相关信息块\n\n### 检索到的信息预览\n{chunks_preview}", "正在生成答案..."

                    # 生成答案
                    background_chunks = "\n\n".join([f"[相关信息 {i + 1}]: {result['chunk']}"
                                                     for i, result in enumerate(search_results)])

                    system_prompt = VERILOG_SYSTEM_PROMPT
                    if use_table_format:
                        system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"

                    user_prompt = f"""
                    {enhanced_question}

                    背景信息：
                    {background_chunks}

                    请基于以上背景信息和对话历史回答用户的问题。
                    """

                    response = client.chat.completions.create(
                        model=Config.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    current_answer = response.choices[0].message.content.strip()
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n检索完成，已生成答案\n\n### 检索到的内容\n{chunks_detail}", current_answer

            except Exception as e:
                error_msg = f"检索过程中出错: {str(e)}"
                yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n{error_msg}", f"检索过程中出错: {str(e)}"
                current_answer = f"检索过程中出错: {str(e)}"

        # 检索完成后，如果有搜索结果，可以考虑合并知识
        if search_future and search_future.done():
            search_result = search_future.result() or "未找到相关网络信息"

            # 如果同时有搜索结果和本地检索结果，可以考虑合并
            if search_result and current_answer and current_answer not in ["正在分析您的问题...",
                                                                           "本地知识库中未找到相关信息。"]:
                status_text = "正在合并联网搜索和知识库结果..."
                if multi_hop:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 推理状态\n{status_text}", current_answer
                else:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n{status_text}", current_answer

                # 合并结果
                system_prompt = "你是一名硬件工程师，请整合网络资料和本地代码库，编写 Verilog 模块。"

                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"

                user_prompt = f"""
                {enhanced_question}

                网络搜索结果：{search_result}

                本地知识库分析：{current_answer}

                请根据以上信息和对话历史，提供一个综合的回答。确保使用Markdown表格来呈现适合表格形式的信息。
                """

                try:
                    response = client.chat.completions.create(
                        model="qwen-plus",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    combined_answer = response.choices[0].message.content.strip()

                    final_status = "已整合联网和知识库结果"
                    if multi_hop:
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成多跳推理分析，检索到的内容已在上方显示\n\n### 综合分析\n{final_status}"
                    else:
                        # 获取之前检索到的内容
                        chunks_info = "".join(
                            [part.split("### 检索到的内容\n")[-1] if "### 检索到的内容\n" in part else "" for part in
                             search_display.split("### 联网搜索结果")])
                        if not chunks_info.strip():
                            chunks_info = "检索内容已在上方显示"
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成向量检索分析\n\n### 检索到的内容\n{chunks_info}\n\n### 综合分析\n{final_status}"

                    yield final_display, combined_answer
                except Exception as e:
                    # 如果合并失败，使用现有答案
                    error_status = f"合并结果失败: {str(e)}"
                    if multi_hop:
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成多跳推理分析，检索到的内容已在上方显示\n\n### 综合分析\n{error_status}"
                    else:
                        # 获取之前检索到的内容
                        chunks_info = "".join(
                            [part.split("### 检索到的内容\n")[-1] if "### 检索到的内容\n" in part else "" for part in
                             search_display.split("### 联网搜索结果")])
                        if not chunks_info.strip():
                            chunks_info = "检索内容已在上方显示"
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成向量检索分析\n\n### 检索到的内容\n{chunks_info}\n\n### 综合分析\n{error_status}"

                    yield final_display, current_answer

    except Exception as e:
        error_msg = f"处理失败：{str(e)}\n{traceback.format_exc()}"
        yield f"### 错误信息\n{error_msg}", f"处理您的问题时遇到错误：{str(e)}"


# 添加处理函数，批量上传文件到指定知识库
def batch_upload_to_kb(file_objs: List, kb_name: str) -> str:
    """批量上传文件到指定知识库并进行处理"""
    print(f"👉 [Debug] 收到上传请求！KB名称: {kb_name}, 文件数量: {len(file_objs) if file_objs else 0}", flush=True)
    try:
        if not kb_name or not kb_name.strip():
            return "错误：未指定知识库"

        # 确保知识库目录存在
        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir, exist_ok=True)

        if not file_objs or len(file_objs) == 0:
            return "错误：未选择任何文件"

        return process_and_index_files(file_objs, kb_name)
    except Exception as e:
        return f"上传文件到知识库失败: {str(e)}"


# Gradio 界面 - 修改为支持多知识库
custom_css = """
.web-search-toggle .form { display: flex !important; align-items: center !important; }
.web-search-toggle .form > label { order: 2 !important; margin-left: 10px !important; }
.web-search-toggle .checkbox-wrap { order: 1 !important; background: #d4e4d4 !important; border-radius: 15px !important; padding: 2px !important; width: 50px !important; height: 28px !important; }
.web-search-toggle .checkbox-wrap .checkbox-container { width: 24px !important; height: 24px !important; transition: all 0.3s ease !important; }
.web-search-toggle input:checked + .checkbox-wrap { background: #2196F3 !important; }
.web-search-toggle input:checked + .checkbox-wrap .checkbox-container { transform: translateX(22px) !important; }
#search-results { max-height: 400px; overflow-y: auto; border: 1px solid #2196F3; border-radius: 5px; padding: 10px; background-color: #e7f0f9; }
#question-input { border-color: #2196F3 !important; }
#answer-output { background-color: #f0f7f0; border-color: #2196F3 !important; max-height: 400px; overflow-y: auto; }
.submit-btn { background-color: #2196F3 !important; border: none !important; }
.reasoning-steps { background-color: #f0f7f0; border: 1px dashed #2196F3; padding: 10px; margin-top: 10px; border-radius: 5px; }
.loading-spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(33, 150, 243, 0.3); border-radius: 50%; border-top-color: #2196F3; animation: spin 1s ease-in-out infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.stream-update { animation: fade 0.5s ease-in-out; }
@keyframes fade { from { background-color: rgba(33, 150, 243, 0.1); } to { background-color: transparent; } }
.status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold; }
.status-processing { background-color: #e3f2fd; color: #1565c0; border-left: 4px solid #2196F3; }
.status-success { background-color: #e8f5e9; color: #2e7d32; border-left: 4px solid #4CAF50; }
.status-error { background-color: #ffebee; color: #c62828; border-left: 4px solid #f44336; }
.multi-hop-toggle .form { display: flex !important; align-items: center !important; }
.multi-hop-toggle .form > label { order: 2 !important; margin-left: 10px !important; }
.multi-hop-toggle .checkbox-wrap { order: 1 !important; background: #d4e4d4 !important; border-radius: 15px !important; padding: 2px !important; width: 50px !important; height: 28px !important; }
.multi-hop-toggle .checkbox-wrap .checkbox-container { width: 24px !important; height: 24px !important; transition: all 0.3s ease !important; }
.multi-hop-toggle input:checked + .checkbox-wrap { background: #4CAF50 !important; }
.multi-hop-toggle input:checked + .checkbox-wrap .checkbox-container { transform: translateX(22px) !important; }
.kb-management { border: 1px solid #2196F3; border-radius: 5px; padding: 15px; margin-bottom: 15px; background-color: #f0f7ff; }
.kb-selector { margin-bottom: 10px; }
/* 缩小文件上传区域高度 */
.compact-upload {
    margin-bottom: 10px;
}

.file-upload.compact {
    padding: 10px;  /* 减小内边距 */
    min-height: 120px; /* 减小最小高度 */
    margin-bottom: 10px;
}

/* 优化知识库内容显示区域 */
.kb-files-list {
    height: 400px;
    overflow-y: auto;
}

/* 确保右侧列有足够空间 */
#kb-files-group {
    height: 100%;
    display: flex;
    flex-direction: column;
}
.kb-files-list { max-height: 250px; overflow-y: auto; border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin-top: 10px; background-color: #f9f9f9; }
#kb-management-container {
    max-width: 800px !important;
    margin: 0 !important; /* 移除自动边距，靠左对齐 */
    margin-left: 20px !important; /* 添加左边距 */
}
.container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.file-upload {
    border: 2px dashed #2196F3;
    padding: 15px;
    border-radius: 10px;
    background-color: #f0f7ff;
    margin-bottom: 15px;
}
.tabs.tab-selected {
    background-color: #e3f2fd;
    border-bottom: 3px solid #2196F3;
}
.group {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 15px;
    background-color: #fafafa;
}

/* 添加更多针对知识库管理页面的样式 */
#kb-controls, #kb-file-upload, #kb-files-group {
    width: 100% !important;
    max-width: 800px !important;
    margin-right: auto !important;
}

/* 修改Gradio默认的标签页样式以支持左对齐 */
.tabs > .tab-nav > button {
    flex: 0 1 auto !important; /* 修改为不自动扩展，只占用必要空间 */
}
.tabs > .tabitem {
    padding-left: 0 !important; /* 移除左边距，使内容靠左 */
}
/* 对于首页的顶部标题部分 */
#app-container h1, #app-container h2, #app-container h3, 
#app-container > .prose {
    text-align: left !important;
    padding-left: 20px !important;
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="gray",
    text_size="lg",
    spacing_size="md",
    radius_size="md"
)

# 添加简单的JavaScript，通过html组件实现
js_code = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // 当页面加载完毕后，找到提交按钮，并为其添加点击事件
    const observer = new MutationObserver(function(mutations) {
        // 找到提交按钮
        const submitButton = document.querySelector('button[data-testid="submit"]');
        if (submitButton) {
            submitButton.addEventListener('click', function() {
                // 找到检索标签页按钮并点击它
                setTimeout(function() {
                    const retrievalTab = document.querySelector('[data-testid="tab-button-retrieval-tab"]');
                    if (retrievalTab) retrievalTab.click();
                }, 100);
            });
            observer.disconnect(); // 一旦找到并设置事件，停止观察
        }
    });

    // 开始观察文档变化
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
"""

with gr.Blocks(title="Verilog 代码生成助手", theme=custom_theme, css=custom_css, elem_id="app-container") as demo:
    with gr.Column(elem_id="header-container"):
        gr.Markdown("""
        # ⚡ Verilog 代码生成助手
        **基于 RAG 的硬件开发 Copilot** 上传你的 Verilog/SystemVerilog 代码库，通过语义检索和推理，生成高质量的可综合代码。
        """)

    # 添加JavaScript脚本
    gr.HTML(js_code, visible=False)

    # 使用State来存储对话历史
    chat_history_state = gr.State([])

    # 创建标签页
    with gr.Tabs() as tabs:
        # 知识库管理标签页
        with gr.TabItem("知识库管理"):
            with gr.Row():
                # 左侧列：控制区
                with gr.Column(scale=1, min_width=400):
                    gr.Markdown("### 📚 知识库管理与构建")

                    with gr.Row(elem_id="kb-controls"):
                        with gr.Column(scale=1):
                            new_kb_name = gr.Textbox(
                                label="新知识库名称",
                                placeholder="输入新知识库名称",
                                lines=1
                            )
                            create_kb_btn = gr.Button("创建知识库", variant="primary", scale=1)

                        with gr.Column(scale=1):
                            current_kbs = get_knowledge_bases()
                            kb_dropdown = gr.Dropdown(
                                label="选择知识库",
                                choices=current_kbs,
                                value=DEFAULT_KB if DEFAULT_KB in current_kbs else (
                                    current_kbs[0] if current_kbs else None),
                                elem_classes="kb-selector"
                            )

                            with gr.Row():
                                refresh_kb_btn = gr.Button("刷新列表", size="sm", scale=1)
                                delete_kb_btn = gr.Button("删除知识库", size="sm", variant="stop", scale=1)

                    kb_status = gr.Textbox(label="知识库状态", interactive=False, placeholder="选择或创建知识库")

                    with gr.Group(elem_id="kb-file-upload", elem_classes="compact-upload"):
                        gr.Markdown("### 📄 上传文件到知识库")
                        file_upload = gr.File(
                            label="选择文件（支持多选TXT/PDF）",
                            type="file",
                            file_types=[".txt", ".pdf",".v",".sv"],
                            file_count="multiple",
                            elem_classes="file-upload compact"
                        )
                        upload_status = gr.Textbox(label="上传状态", interactive=False, placeholder="上传后显示状态")

                    kb_select_for_chat = gr.Dropdown(
                        label="为对话选择知识库",
                        choices=current_kbs,
                        value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                        visible=False  # 隐藏，仅用于同步
                    )

                with gr.Column(scale=1, min_width=400):
                    with gr.Group(elem_id="kb-files-group"):
                        gr.Markdown("### 📋 知识库内容")
                        kb_files_list = gr.Markdown(
                            value="选择知识库查看文件...",
                            elem_classes="kb-files-list"
                        )

                # 用于对话界面的知识库选择器
                kb_select_for_chat = gr.Dropdown(
                    label="为对话选择知识库",
                    choices=current_kbs,
                    value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                    visible=False  # 隐藏，仅用于同步
                )

        # 对话交互标签页
        with gr.TabItem("对话交互"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 对话设置")

                    kb_dropdown_chat = gr.Dropdown(
                        label="选择知识库进行对话",
                        choices=current_kbs,
                        value=DEFAULT_KB if DEFAULT_KB in current_kbs else (current_kbs[0] if current_kbs else None),
                    )

                    with gr.Row():
                        web_search_toggle = gr.Checkbox(
                            label="🌐 启用联网搜索",
                            value=True,
                            info="获取最新医疗动态",
                            elem_classes="web-search-toggle"
                        )
                        table_format_toggle = gr.Checkbox(
                            label="📊 表格格式输出",
                            value=True,
                            info="使用Markdown表格展示结构化回答",
                            elem_classes="web-search-toggle"
                        )

                    multi_hop_toggle = gr.Checkbox(
                        label="🔄 启用多跳推理",
                        value=False,
                        info="使用高级多跳推理机制（较慢但更全面）",
                        elem_classes="multi-hop-toggle"
                    )

                    with gr.Accordion("显示检索进展", open=False):
                        search_results_output = gr.Markdown(
                            label="检索过程",
                            elem_id="search-results",
                            value="等待提交问题..."
                        )

                with gr.Column(scale=3):
                    gr.Markdown("### 💬 对话历史")
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        label="对话历史",
                        height=550
                    )

            with gr.Row():
                question_input = gr.Textbox(
                    label="输入需要生成verilog的要求",
                    placeholder="例如：写一个带有异步复位的 8 位计数器",
                    lines=2,
                    elem_id="question-input"
                )

            with gr.Row(elem_classes="submit-row"):
                submit_btn = gr.Button("提交问题", variant="primary", elem_classes="submit-btn")
                clear_btn = gr.Button("清空输入", variant="secondary")
                clear_history_btn = gr.Button("清空对话历史", variant="secondary", elem_classes="clear-history-btn")

            # 状态显示框
            status_box = gr.HTML(
                value='<div class="status-box status-processing">准备就绪，等待您的问题</div>',
                visible=True
            )

            gr.Examples(
                examples=[
                    ["写一个带有异步复位的 8 位计数器"],
                    ["生成一个 AXI4-Lite Slave 接口模板"],
                    ["如何实现跨时钟域的单比特信号同步？"],
                    ["写一个用于视频处理的 Line Buffer"],
                    ["解释一下这段代码中的状态机逻辑"]
                ],
                inputs=question_input,
                label="示例问题（点击尝试）"
            )


    # 创建知识库函数
    def create_kb_and_refresh(kb_name):
        result = create_knowledge_base(kb_name)
        kbs = get_knowledge_bases()
        # 更新两个下拉菜单
        return result, gr.update(choices=kbs, value=kb_name if "创建成功" in result else None), gr.update(choices=kbs,
                                                                                                          value=kb_name if "创建成功" in result else None)


    # 刷新知识库列表
    def refresh_kb_list():
        kbs = get_knowledge_bases()
        # 更新两个下拉菜单
        return gr.update(choices=kbs, value=kbs[0] if kbs else None), gr.update(choices=kbs,
                                                                                value=kbs[0] if kbs else None)


    # 删除知识库
    def delete_kb_and_refresh(kb_name):
        result = delete_knowledge_base(kb_name)
        kbs = get_knowledge_bases()
        # 更新两个下拉菜单
        return result, gr.update(choices=kbs, value=kbs[0] if kbs else None), gr.update(choices=kbs,
                                                                                        value=kbs[0] if kbs else None)


    # 更新知识库文件列表
    def update_kb_files_list(kb_name):
        if not kb_name:
            return "未选择知识库"

        files = get_kb_files(kb_name)
        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        has_index = os.path.exists(os.path.join(kb_dir, "semantic_chunk.index"))

        if not files:
            files_str = "知识库中暂无文件"
        else:
            files_str = "**文件列表:**\n\n" + "\n".join([f"- {file}" for file in files])

        index_status = "\n\n**索引状态:** " + ("✅ 已建立索引" if has_index else "❌ 未建立索引")

        return f"### 知识库: {kb_name}\n\n{files_str}{index_status}"


    # 同步知识库选择 - 管理界面到对话界面
    def sync_kb_to_chat(kb_name):
        return gr.update(value=kb_name)


    # 同步知识库选择 - 对话界面到管理界面
    def sync_chat_to_kb(kb_name):
        return gr.update(value=kb_name), update_kb_files_list(kb_name)


    # 处理文件上传到指定知识库
    def process_upload_to_kb(files, kb_name):
        if not kb_name:
            return "错误：未选择知识库",None

        result = batch_upload_to_kb(files, kb_name)
        # 更新知识库文件列表
        files_list = update_kb_files_list(kb_name)
        return result, files_list


    # 知识库选择变化时
    def on_kb_change(kb_name):
        if not kb_name:
            return "未选择知识库", "选择知识库查看文件..."

        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        has_index = os.path.exists(os.path.join(kb_dir, "semantic_chunk.index"))
        status = f"已选择知识库: {kb_name}" + (" (已建立索引)" if has_index else " (未建立索引)")

        # 更新文件列表
        files_list = update_kb_files_list(kb_name)

        return status, files_list


    # 创建知识库按钮功能
    create_kb_btn.click(
        fn=create_kb_and_refresh,
        inputs=[new_kb_name],
        outputs=[kb_status, kb_dropdown, kb_dropdown_chat]
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=[],
        outputs=[new_kb_name]
    )

    # 刷新知识库列表按钮功能
    refresh_kb_btn.click(
        fn=refresh_kb_list,
        inputs=[],
        outputs=[kb_dropdown, kb_dropdown_chat]
    )

    # 删除知识库按钮功能
    delete_kb_btn.click(
        fn=delete_kb_and_refresh,
        inputs=[kb_dropdown],
        outputs=[kb_status, kb_dropdown, kb_dropdown_chat]
    ).then(
        fn=update_kb_files_list,
        inputs=[kb_dropdown],
        outputs=[kb_files_list]
    )

    # 知识库选择变化时 - 管理界面
    kb_dropdown.change(
        fn=on_kb_change,
        inputs=[kb_dropdown],
        outputs=[kb_status, kb_files_list]
    ).then(
        fn=sync_kb_to_chat,
        inputs=[kb_dropdown],
        outputs=[kb_dropdown_chat]
    )

    # 知识库选择变化时 - 对话界面
    kb_dropdown_chat.change(
        fn=sync_chat_to_kb,
        inputs=[kb_dropdown_chat],
        outputs=[kb_dropdown, kb_files_list]
    )

    # 处理文件上传
    file_upload.upload(
        fn=process_upload_to_kb,
        inputs=[file_upload, kb_dropdown],
        outputs=[upload_status, kb_files_list]
    )

    # 清空输入按钮功能
    clear_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=[question_input]
    )


    # 清空对话历史按钮功能
    def clear_history():
        return [], []


    clear_history_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[chatbot, chat_history_state]
    )


    # 提交按钮 - 开始流式处理
    def update_status(is_processing=True, is_error=False):
        if is_processing:
            return '<div class="status-box status-processing">正在处理您的问题...</div>'
        elif is_error:
            return '<div class="status-box status-error">处理过程中出现错误</div>'
        else:
            return '<div class="status-box status-success">回答已生成完毕</div>'


    # 处理问题并更新对话历史
    def process_and_update_chat(question, kb_name, use_search, use_table_format, multi_hop, chat_history):
        if not question.strip():
            return chat_history, update_status(False, True), "等待提交问题..."

        try:
            # 首先更新聊天界面，显示用户问题
            chat_history.append([question, "正在思考..."])
            yield chat_history, update_status(True), f"开始处理您的问题，使用知识库: {kb_name}..."

            # 用于累积检索状态和答案
            last_search_display = ""
            last_answer = ""

            # 使用生成器进行流式处理
            for search_display, answer in process_question_with_reasoning(question, kb_name, use_search,
                                                                          use_table_format, multi_hop,
                                                                          chat_history[:-1]):
                # 更新检索状态和答案
                last_search_display = search_display
                last_answer = answer

                # 更新聊天历史中的最后一条（当前的回答）
                if chat_history:
                    chat_history[-1][1] = answer
                    yield chat_history, update_status(True), search_display

            # 处理完成，更新状态
            yield chat_history, update_status(False), last_search_display

        except Exception as e:
            # 发生错误时更新状态和聊天历史
            error_msg = f"处理问题时出错: {str(e)}"
            if chat_history:
                chat_history[-1][1] = error_msg
            yield chat_history, update_status(False, True), f"### 错误\n{error_msg}"


    # 连接提交按钮
    submit_btn.click(
        fn=process_and_update_chat,
        inputs=[question_input, kb_dropdown_chat, web_search_toggle, table_format_toggle, multi_hop_toggle,
                chat_history_state],
        outputs=[chatbot, status_box, search_results_output],
        queue=True
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=[],
        outputs=[question_input]
    ).then(
        fn=lambda h: h,  # 更新state
        inputs=[chatbot],
        outputs=[chat_history_state]
    )

    # 支持Enter键提交
    question_input.submit(
        fn=process_and_update_chat,
        inputs=[question_input, kb_dropdown_chat, web_search_toggle, table_format_toggle, multi_hop_toggle,
                chat_history_state],
        outputs=[chatbot, status_box, search_results_output],
        queue=True
    ).then(
        fn=lambda: "",  # 清空输入框
        inputs=[],
        outputs=[question_input]
    ).then(
        fn=lambda h: h,  # 更新state
        inputs=[chatbot],
        outputs=[chat_history_state]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)