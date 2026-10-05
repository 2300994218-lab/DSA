# 新闻来源与 AI 搜索链路

V1.1.0 在「设置 → 新闻来源」中提供统一的来源管理和测试页面。启用来源后，RSS、NewsNow 和 AKShare 资讯会进入 `SearchService`，因此 AI 问股、个股分析、市场分析以及 Agent 新闻检索都可以使用这些结果；Tavily、Brave 等原有搜索提供方仍按原优先级工作。

页面提供两种测试：

- 单源测试：以 dry-run 方式请求一个已启用来源，显示抓取数量和样例，不写入资讯表。
- 完整链路测试：依次请求所有已启用来源，显示每个来源的成功、失败和抓取数量，同样不写入资讯表。

内置来源包括：

- 中国新闻网财经与滚动 RSS。
- 财联社电报、东方财富 7×24、同花顺 7×24，通过 AKShare 的 `stock_info_global_cls`、`stock_info_global_em`、`stock_info_global_ths` 接入。
- 金十数据通过已有 NewsNow 适配接入。
- 新浪财经 7×24、富途快讯、东方财富财经早餐作为免费补充来源。
- SEC、Federal Reserve、CNBC 等公开 RSS，覆盖美股与宏观新闻。

AKShare 来源的 URL 使用内部标识（例如 `akshare://global_cls`），只能选择内置模板，不能调用任意 Python 函数。AKShare 请求使用现有 `NEWS_INTEL_FETCH_TIMEOUT_SEC` 超时；单个来源失败会记录在测试结果中，不会阻断其他来源或整个分析任务。

新部署默认不自动启用全部模板。请在页面中逐个添加并测试需要的来源。NewsNow 的服务地址仍由 `NEWSNOW_BASE_URL` 控制，RSS 地址沿用现有 URL 安全校验和重定向限制。
