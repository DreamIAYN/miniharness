# Security Policy

## 报告漏洞

请不要在公开 Issue 中提交安全漏洞、API Key、admin token、数据库文件或其他敏感信息。

优先使用 GitHub 的私密漏洞报告：

<https://github.com/DreamIAYN/miniharness/security/advisories/new>

如果无法使用该入口，请联系仓库维护者，并在主题中注明 `[SECURITY]`。

## 敏感信息

- `.env` 已被 `.gitignore` 忽略，不能提交。
- `data/`（SQLite、workspace、调用正文）不能提交。
- 生产环境必须设置强随机 `HARNESS_ADMIN_TOKEN`。
- 用户 API Key（`hx_...`）只在创建时显示一次；如果泄露，立即吊销并重建。
- 如果模型 API Key 或 admin token 曾进入 Git 历史，请立即轮换，不能只从最新提交删除。
- 本项目默认的 `local` 沙箱只做资源隔离，不是安全边界；执行不可信代码请使用 Docker 沙箱。

## 支持范围

安全修复默认针对 `main` 分支的最新版本。
