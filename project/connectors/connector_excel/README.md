# Excel与CSV Connector

独立Manifest名称connector.excel，契约版本1，入口adapter:load。支持XLSX与CSV单文件/目录，auto通过Manifest的.xlsx/.csv识别。复制独立目录、配置允许列表并重启即可接入，不修改Core或其他插件。

XLSX复用现有 `docparse.office.structured_parse(path, options)`；worker只是独立解析环境的进程桥接，依赖openpyxl>=3.1。设置docparse_repo及docparse_python，options.parser_python可覆盖解释器，options.timeout_s限制解析期限。不自动安装依赖、不执行宏或任意公式；没有公式缓存时保留不可用/复核信息，不猜结果。原始值、显示格式、日期、空格、工作表、表头与行列位置进入公开结构。旧.xls不在专用声明范围。

CSV使用标准库csv，在宿主解析，不需openpyxl或新模型。默认严格尝试utf-8-sig、gb18030，BOM标识的UTF-16/32另行处理；GB18030覆盖常见GBK。options.encoding可显式指定，options.delimiter可指定分隔符，options.header_rows默认1、options.max_cells默认200000。未知或损坏编码、列数不一致、表头范围错误明确报错，不用errors=ignore。保留字符串小数、零、空字段、引号与字段内换行；csv_record_lines保留记录对应原行。

两种格式均输出Document.segments.meta中的table_data/location等既有结构，page=0、bbox为空，不伪造物理页。Core按内容类型统一处理，不需插件新增问答接口。单文件故障进入ConnectorIssue，目录允许部分成功；禁用由Registry返回CONNECTOR_DISABLED。宿主和解析进程都是受信本地代码，不是恶意文件沙箱。

公开字段、生命周期、错误和测试入口见[插件协议](../../docs/RAG_PLUGIN_CONTRACTS.md)，B版最新人工结果见[验收报告](../../docs/testing/FINAL_ACCEPTANCE.md)。原始来源/第三方许可须保留，docparse整体再分发由团队确认。AI示例或本地代码存在不等于其他成员独立贡献与Review。
