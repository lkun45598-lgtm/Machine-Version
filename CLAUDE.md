# CLAUDE.md

## 环境
- conda 环境：`pytorch`
- 运行方式：`conda run -n pytorch python xxx.py` 或 `/home/lz/miniconda3/envs/pytorch/bin/python xxx.py`

## 项目结构
- 按周分文件夹：`week1/`、`week2/`、...
- 每周内部按模块组织：`src/`（代码）、`data/`（输入与输出图像）、`report/`（实验报告）、`reference/`（课件与模板）
- 根目录 `README.md` 维护各周实验汇总表格

## 代码规范
- 路径全部用 `Path(__file__).resolve().parent` 锚定，禁止使用相对当前工作目录的字符串路径
- OpenCV 默认 BGR 通道，传给 matplotlib 显示前必须 `cv2.cvtColor(..., COLOR_BGR2RGB)`
- 标准测试图优先用 `skimage.data` 提供的图像（如 `coffee()`、`astronaut()`），保证可复现
- 实验报告 Markdown 中的图片链接用相对 .md 文件位置的路径（如 `../data/outputs/xxx.png`），正文中提及的文件路径用项目相对路径（如 `data/outputs/xxx.png`）
- docx 由 `generate_lab_report_docx.py` 从对应 .md 一键生成，不手动改 docx

## 实验报告写作
- 不引用"PPT"、"课件"、"实验任务书"等来源，以独立完整文档的口吻陈述事实
- 不写"按 XX 要求做了 YY"这类元说明
- 不在正文中加粗强调（`**...**`），保持学术报告语气
- 个人信息（姓名 / 学号 / 专业 / 班级）放在 4 列表格中，紧随 "实 验 报 告" 大标题

## GitHub
- 仓库：`github.com/lkun45598-lgtm/Machine-Version`
- 每周实验完成后推送，commit message 格式：`Week X: 实验内容简述`
- 推送时 token 通过临时 URL 传递，不写入 git config（避免泄漏到 `.git/config` 或 `branch.*.remote`）
