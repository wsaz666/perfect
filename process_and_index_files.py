def process_and_index_files(file_objs: List, kb_name: str = DEFAULT_KB) -> str:
    print("！！！进入了文件处理函数！！！", flush=True)
    """处理并索引文件到指定的知识库"""
    # 确保知识库目录存在
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    os.makedirs(kb_dir, exist_ok=True)

    # 设置临时处理文件路径
    semantic_chunk_output = os.path.join(OUTPUT_DIR, "semantic_chunk_output.json")
    semantic_chunk_vector = os.path.join(OUTPUT_DIR, "semantic_chunk_vector.json")

    # 设置知识库索引文件路径
    semantic_chunk_index = os.path.join(kb_dir, "semantic_chunk.index")
    semantic_chunk_metadata = os.path.join(kb_dir, "semantic_chunk_metadata.json")

    new_chunks = []  # 改名：这里存放本次新处理的分块
    error_messages = []

    try:
        if not file_objs or len(file_objs) == 0:
            return "错误：没有选择任何文件"

        print(f"开始处理 {len(file_objs)} 个文件，目标知识库: {kb_name}...")

        # 1. 多线程处理文件读取和初步切分
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_file = {executor.submit(process_single_file, file_obj.name): file_obj for file_obj in file_objs}
            for future in as_completed(future_to_file):
                result = future.result()
                file_obj = future_to_file[future]
                file_name = file_obj.name

                if isinstance(result, str) and result.startswith("处理文件"):
                    error_messages.append(result)
                    print(result)
                    continue

                # 检查结果是否为有效文本
                if not result or not isinstance(result, str) or len(result.strip()) == 0:
                    error_messages.append(f"文件 {file_name} 处理后内容为空")
                    print(f"警告: 文件 {file_name} 处理后内容为空")
                    continue

                print(f"对文件 {file_name} 进行语义分块...")
                # 调用你的 semantic_chunk 函数
                chunks = semantic_chunk(result)

                if not chunks or len(chunks) == 0:
                    error_messages.append(f"文件 {file_name} 无法生成任何分块")
                    print(f"警告: 文件 {file_name} 无法生成任何分块")
                    continue

                # 将处理后的源文件复制保存到知识库目录
                file_basename = os.path.basename(file_name)
                dest_file_path = os.path.join(kb_dir, file_basename)
                try:
                    shutil.copy2(file_name, dest_file_path)
                    print(f"已将文件 {file_basename} 复制到知识库 {kb_name}")
                except Exception as e:
                    print(f"复制文件到知识库失败: {str(e)}")

                # 为这一批 chunks 临时打上文件名标签，方便后续生成 ID
                for c in chunks:
                    c["metadata"] = {"source": file_basename}

                new_chunks.extend(chunks)
                print(f"文件 {file_name} 处理完成，生成 {len(chunks)} 个分块")

        if not new_chunks:
            return "所有文件处理失败或内容为空\n" + "\n".join(error_messages)

        # 2. 清洗文本并生成唯一 ID (核心修改)
        valid_chunks = []
        import hashlib  # 引入 hashlib

        for chunk in new_chunks:
            # 深度清理文本
            clean_chunk_text = clean_text(chunk["chunk"])
            source_file = chunk.get("metadata", {}).get("source", "unknown")

            # 检查清理后的文本是否有效
            if clean_chunk_text and 1 <= len(clean_chunk_text) <= 8000:
                chunk["chunk"] = clean_chunk_text

                # 🟢 修改点：生成唯一 Hash ID (文件名+内容)
                # 解决了分批上传 ID 冲突和重置的问题
                unique_str = f"{source_file}_{clean_chunk_text}"
                chunk["id"] = hashlib.md5(unique_str.encode('utf-8')).hexdigest()

                valid_chunks.append(chunk)

            elif len(clean_chunk_text) > 8000:
                # 截断处理
                chunk["chunk"] = clean_chunk_text[:8000]

                # 截断后同样生成 ID
                unique_str = f"{source_file}_{chunk['chunk']}"
                chunk["id"] = hashlib.md5(unique_str.encode('utf-8')).hexdigest()

                valid_chunks.append(chunk)
                print(f"警告: 分块过长已被截断，源文件: {source_file}")
            else:
                print(f"警告: 跳过无效分块")

        if not valid_chunks:
            return "所有生成的分块内容无效或为空\n" + "\n".join(error_messages)

        print(f"本次新增 {len(valid_chunks)} 个有效分块")

        # 3. 增量合并保存 (核心修改)
        # 读取已有的 JSON 数据，防止覆盖旧数据
        final_all_chunks = []
        if os.path.exists(semantic_chunk_output):
            try:
                with open(semantic_chunk_output, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                    if isinstance(old_data, list):
                        final_all_chunks = old_data
            except Exception as e:
                print(f"读取旧数据失败，将重新创建: {e}")

        # 使用字典去重合并：{id: chunk_data}
        # 如果 ID 相同（内容+文件名相同），新数据会覆盖旧数据
        chunk_map = {item["id"]: item for item in final_all_chunks}

        for item in valid_chunks:
            chunk_map[item["id"]] = item

        # 转回列表
        final_all_chunks = list(chunk_map.values())

        # 保存合并后的完整列表
        with open(semantic_chunk_output, 'w', encoding='utf-8') as json_file:
            json.dump(final_all_chunks, json_file, ensure_ascii=False, indent=4)
        print(f"语义分块完成，当前库中总计 {len(final_all_chunks)} 个分块。路径: {semantic_chunk_output}")

        # 4. 向量化 (注意：这里我们对整个库进行向量化，保证索引完整)
        # 如果数据量巨大，后续可优化为只向量化新增部分，但目前全量最稳
        print(f"开始向量化所有 {len(final_all_chunks)} 个分块...")
        vectorize_file(final_all_chunks, semantic_chunk_vector)
        print(f"语义分块向量化完成: {semantic_chunk_vector}")

        # 验证向量文件
        try:
            with open(semantic_chunk_vector, 'r', encoding='utf-8') as f:
                vector_data = json.load(f)

            if not vector_data or len(vector_data) == 0:
                return f"向量化失败: 生成的向量文件为空\n" + "\n".join(error_messages)

            if 'vector' not in vector_data[0]:
                return f"向量化失败: 数据中缺少向量字段\n" + "\n".join(error_messages)

            print(f"成功生成 {len(vector_data)} 个向量")
        except Exception as e:
            return f"读取向量文件失败: {str(e)}\n" + "\n".join(error_messages)

        # 5. 构建索引
        print(f"开始为知识库 {kb_name} 构建索引...")
        build_faiss_index(semantic_chunk_vector, semantic_chunk_index, semantic_chunk_metadata)
        print(f"知识库 {kb_name} 索引构建完成: {semantic_chunk_index}")

        status = f"知识库 {kb_name} 更新成功！本次新增 {len(valid_chunks)} 个分块，库中总计 {len(final_all_chunks)} 个。\n"
        if error_messages:
            status += "以下文件处理过程中出现问题：\n" + "\n".join(error_messages)
        return status

    except Exception as e:
        error = f"知识库 {kb_name} 索引构建过程中出错：{str(e)}"
        print(error)
        traceback.print_exc()
        return error + "\n" + "\n".join(error_messages)