# Windows 本地部署

项目源码已放在当前目录，使用项目目录内的 `.venv`、`.runtime`、`.cache` 和 `logs` 保存运行环境与运行数据；脚本优先使用目录内运行时，迁移到没有 `.runtime` 的机器时会回退到系统 Python/Node。Docker Desktop Linux 引擎当前未运行，因此已完成 Windows 本地部署。

首次安装或重新安装（PowerShell）：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup-windows.ps1
```

脚本会安装 `requirements.txt`、构建 `apps/dsa-web` 并验证 `static` 静态资源。启动后端和 WebUI：

```powershell
.\scripts\start-windows.cmd
```

启动脚本在后台运行服务。首次启动导入依赖和缓存可能需要十几秒；查看日志：

```powershell
Get-Content .\logs\web.stdout.log -Wait
Get-Content .\logs\web.stderr.log -Wait
```

浏览器访问 <http://127.0.0.1:8000>，健康检查为 <http://127.0.0.1:8000/api/health>。停止服务：

```powershell
.\scripts\stop-windows.cmd
```

把 `.env` 中至少一个 LLM 配置（例如 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LITELLM_MODEL`，或项目支持的其他 provider）填好后，才能执行 AI 分析；免费行情源可直接使用，通知渠道按需配置。当前 `.env` 已关闭启动即分析和自动前端重建，避免服务启动时误跑任务。

如需局域网访问，可在 `.env` 设置 `WEBUI_HOST=0.0.0.0`，并先启用 `ADMIN_AUTH_ENABLED=true` 设置密码；Windows 防火墙需自行放行对应端口。

Docker 方式仍可参考 `docs/full-guide.md` 的 Docker 部分；需要先启动 Docker Desktop 的 Linux 容器引擎。

