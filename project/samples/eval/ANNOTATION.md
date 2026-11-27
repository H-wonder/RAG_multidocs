# P1 评测集与标注说明

`questions.json`原30题保留；`p1_questions.json`共60题，前30题的问题和参考答案保留，后30题覆盖实际PNG/JPG、Word、Excel、多级表头、跨sheet、显式计算和跨来源。所有题属于已见开发/回归样本，annotation_status=PENDING_HUMAN。历史split=dev/test仅为原字段，**不表示本轮未见集**。没有独立人工确认的最终验证集。

## 语料及位置

旧30题使用`project/samples/campus-corpus`的8个逻辑来源（TXT、Markdown、HTML、原生PDF、Word、真实图片）。评测工具实际调用对应Connector，不用`normalized.jsonl`代替OCR。新30题使用`project/tools/rag_samples.py`生成的真实小型二进制测试文件；图片中的金额、日期、状态是我们写入图片的明确字段，不是编造的OCR输出。DOCX段落/表格/图片，XLSX Models/Budget/Merged及原始单元格均可直接核对。

所有策略在一次导入后共用固定活动语料；manifest保存语料/题集SHA-256、源码哈希、collection、完整policy、入库结果。未修改实验中的文件或gold来追求分数。Word/Excel无物理页page=0；图片page=1而无局部bbox。gold的`contains`为预期原文连续摘录，空白归一匹配，不能表达一般语义蕴含。需结合真实原件进行人工确认。

旧30题显式`source_allowlist`限制旧来源cohort，避免新实验工作簿的SVM95%改变原“课程实验SVM数据未提供”的语境。新题在标注中说明所指文档，允许跨文档题使用指定多个来源。这是**有来源约束的开发评测**，不能当全库不加范围的泛化召回结果。所有策略使用同样scope，不能据此声称全库歧义问题已解决。

`reference_answer`为待审核的作者/AI开发标注，不取自生成模型答案。`expected_evidence`含source/page/contains；必要子问题在`required_subquestions`，有明确证据映射时使用`subquestion_evidence`。`automatic_answer_contains`是格式敏感的字面冒烟检查：95%与95.00%可能不相等，不能计作Answer Accuracy。

## 人工复核和冻结流程

1. 复核每题原件、所问范围、单位、数值、真实位置、gold摘录及可回答性；不要按系统结果修改答案。
2. 另由未参与调参的成员构造新题，独立审核并冻结hash；在这之前held-out状态保持PENDING_HUMAN。
3. 冻结模型/语料/策略/环境后执行一次最终验证；如果据此修改代码，该批题转为已见开发样本，另建最终集。
4. 人工结果需真实reviewer，按题ID提交`answer_correct`、`citation_correct`（长度等于实际引用数）、`answer_complete`、`table_numeric_error`布尔值，可另记理由。`run_suite(..., reviews=...)`只计算带reviewer的实际审核；不可伪造身份。原始Prompt/输出/定位需一起核对。

## 指标边界

固定K=3分别统计基础与重排阶段Hit/MRR/Recall，分母只含有gold的题。最终上下文gold覆盖和来源覆盖另算，来源出现不等于满足所有事实。人工Answer Accuracy、Citation Precision、多子问答案完整性、表格数值错误率无审核时均null。库外拒答/误拒/生成故障由已标注answerable与实际status机械计算，其可回答性标签本身仍待复核。

Gold救回只在触发第二轮的可回答题上，按首轮/末轮实际gold新增计算；无效重召回无新增gold，不用算法自己的missing标志自证。Token预算估计与Provider使用量分列。P95为有限样本nearest-rank，顺序运行受缓存及暖机影响，没有随机化顺序/置信区间/大规模统计证明。

Milvus+BGE与Keyword结果分目录，Keyword不含真实向量检索；部分Keyword增强仍调用本地改写模型，不把它称为完全Mock。独立Mock生成验收另有mode字段。历史P0基线采用参考转录夹具，不能与本轮真实解析的绝对指标当严格前后对照。

## 单独OCR挑战集

`p1_ocr_challenge.json`另含12道已见开发挑战，覆盖多页中文/英文、原生/扫描金额、同页混合、相近数值/指标、不同来源与活动版本、库外年份及缺失税率。使用`rag_ocr_acceptance.py`完成后的独立state，PNG当前319.40而JPG仍246.80。题目和参考字段来自自制原件及可核对位置，标签仍PENDING_HUMAN。所有题保留，不按增强优劣筛选；与原60题分开报告。它不是新held-out，不把来源出现等同于税率/Accuracy事实对应。
