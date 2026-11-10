# 自动质量门

目标是**尽量不依赖人工复核**：每个可检测的失败模式必须要么自动修复，要么以机器可读的告警输出。

## 告警码

| code | severity | 含义 | 自动动作 |
|---|---|---|---|
| `repetition` | medium | 输出出现循环/重复 | `presence_penalty=1.5` 重试 |
| `repetition_unresolved` | high | 重试后仍重复 | 升级到 region |
| `truncated` | high | `finish_reason=length` | 降低输出预算重试，或升级 region |
| `empty_output` | high | 空响应 | 升级 region |
| `low_coverage` | medium | 页面墨迹多但输出 <10 字 | 记录（保守触发） |
| `table_parse_failed` | medium | OTSL 转 HTML 失败 | 保留原始 OTSL |
| `field_not_grounded` | high/medium | KIE 字段在 OCR 正文中找不到 | 标记（数字 ≥5 位为 high） |
| `field_missing` | info | KIE 字段为空 | 记录 |
| `kie_invalid` | high | KIE 输出不是合法 JSON | 记录 |
| `region_failed` | medium | 单个版面区域失败 | 丢弃该区域，保留整页 |
| `page_failed` | high | 整页处理异常 | 记录，继续下一页 |
| `native_text` / `ocr_path` | info | 记录路由决策 | — |

只有 `high` 会把文档标为 `needs_review=true`；`medium` 会出现在告警列表但不阻塞。

## 重复检测

复用上游 `repetition_guard.analyze_repetition`，检测三类信号：

- 同一字符连续出现 ≥128 次
- 尾部周期性重复（≥8 次、覆盖 ≥50%）
- 同一行重复（≥8 次、占比 ≥80%）

触发后自动用 `presence_penalty=1.5` 重试一次；分数不低于原结果才采纳。

## KIE 幻觉回查

对每个抽取字段，在 OCR 正文中做字面回查：

- **数字**：去掉标点后按位比对（`1,234.56` ≡ `1234.56`）
- **日期**：识别 `YYYY-MM-DD` / `YY-MM-DD` / `YYYY年MM月DD日`，拆分年月日分别比对；四位年份允许匹配后两位（`2017` ≡ `17`）
- **文本**：去空白后子串匹配

命中失败 → `field_not_grounded`。数字位数 ≥5（金额、单号）判 high，其余 medium。

> 已知局限：语义错误无法通过字面回查发现。例如模型把商品条码当成“发票号码”，数字确实在正文里，会通过检查。这类错误需要业务规则（校验位、跨字段一致性）进一步约束。

## 覆盖率启发式

`ink_ratio` 用 **Otsu 自适应阈值**估计前景占比，而不是固定灰度阈值：

- 固定阈值会被深色纸张/纹理背景骗到（千字文照片曾误报 100% 墨迹）
- 对比度 <40 直接视为空白页

触发条件刻意保守：**输出 <10 字且墨迹 >15%** 才告警。短文档（摘录、标题页、书法）是合法的，宁可漏报也不制造假阳性。

## 调参原则

1. 假阳性比漏报更贵——它会直接增加复核负担，违背“不人工复核”的目标。
2. 质量门只做**可自动决策**的判断；语义正确性交给业务规则和交叉校验。
3. 每次调整都应在 `docparse eval` 的参考集上回归，确认没有把正常文档标成 review。
