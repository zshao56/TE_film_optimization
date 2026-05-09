# Agent Operating Instructions (AI 助手操作指南)

This document contains the core operational rules for any AI agent working on the `TE_film_optimization` repository. The agent must strictly adhere to these protocols.

## 1. 版本控制与代码提交流程 (Version Control & Commits)
- **主动提交机制 (Proactive Commits)**: 当完成一个显著的功能模块（Significant Change）、重构、修复关键 Bug、或者按照用户的要求编写完一整段计划 (Plan) 后，代理必须**主动**将修改提交 (Commit) 到 Git，并推送到远端 (Push)。不要总是等待用户的显式催促。
- **原子化提交 (Atomic Commits)**: 提交信息应遵循常规格式（例如 `feat:`, `fix:`, `docs:`, `refactor:`）。每次提交应该只包含一个逻辑功能的修改，保持历史记录的清晰。
- **状态检查 (Status Check)**: 在修改前后，主动使用 `git status` 确认未跟踪 (untracked) 的文件是否需要被加入追踪，防止遗漏新增的代码文件。

## 2. 依赖管理与环境同步 (Dependency Management)
- **实时更新 (Real-time Updates)**: 如果在编写代码时引入了任何新的第三方库（例如 `torch`, `pandas`, `scipy` 等），代理**必须立刻**修改根目录下的 `requirements.txt` 文件，将新依赖添加进去。
- **杜绝幽灵依赖**: 不允许出现代码中 `import` 了某个新库，但 `requirements.txt` 没有更新的情况。必须保证任何人在拉取仓库后运行 `pip install -r requirements.txt` 就能完美复现环境。

## 3. 规划与文档记录 (Planning & Documentation)
- **日志更新 (Log Updates)**: 所有重大的理论推导、架构改变（如更换网格精度、新增算法模型）都必须实时记录在 `plan.md` 的日志（Log）区域。
- **双语文档 (Bilingual Docs)**: 核心面向用户的说明文档（如 `README.md`）应保持中英双语维护（或提供链接如 `README_zh.md`）。

## 4. 仿真与物理约束 (Simulation & Physics Constraints)
- **网格底线**: 目前系统已通过网格无关性测试，默认的 FDM 求解网格分辨率固定为 `nx=50, ny=50, nz=20`。代理在任何涉及仿真的修改中，不得在未经用户同意的情况下私自降低或过度调高该分辨率。
- **边界条件**: 严格遵守 `database_temperature_difference_protocol.md` 中定义的物理边界。代理在优化目标函数或后处理代码时，禁止改变自然对流或恒温边界的物理本质。
