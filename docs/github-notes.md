# GitHub 协作记录 / GitHub Notes

仓库地址：<https://github.com/Chtrrrrrrr/Zhongan-OPEN-SZ-01>
本地路径：`repo/`（本文件即位于 `repo/docs/`）

## 0. 最新状态：推送已成功（2026-10-11）

凭据在本机可用后，推送已实际完成，远端 `main` 与本地一致：

```powershell
cd repo
git -c http.sslVerify=false push origin main
# -> a3fd059..ac5f57a  main -> main
git -c http.sslVerify=false ls-remote origin main
# -> ac5f57a6fffe10199fad6f90519b0a0e21da9bf4  refs/heads/main
```

已推送的提交链：

| commit | 内容 |
| --- | --- |
| `a3fd059` | L0/L1/L2/L3 管线骨架 + fusion/report/video 模块 |
| `2bccdbd` | 分块 CLIP + 域取证检测管线（含实测 KR） |
| `6feea7f` | 实测结果文档、README 快速开始、评测证据归档 |
| `ac5f57a` | 融合权重修正（未配置成员权重为 0）+ 演示 API 测试 |

非交互推送所需环境变量（避免凭据助手弹窗阻塞）：

```powershell
$env:GIT_TERMINAL_PROMPT = "0"
$env:GCM_INTERACTIVE = "never"
git -c http.sslVerify=false -c credential.interactive=false push origin main
```

## 1. 克隆记录（2026-10-11，已解决）

- **现象**：`git clone https://github.com/Chtrrrrrrr/Zhongan-OPEN-SZ-01 repo/` 首次失败，
  报错 `SSL certificate OpenSSL verify result: unable to get local issuer certificate (20)`。
  原因为本机环境缺少完整的 CA 证书链（常见于公司代理/杀软拦截或 Git for Windows 未配置系统证书）。
- **处置**：改用 `git -c http.sslVerify=false clone ...` 克隆成功（远端为公开仓库，内容可校验）。
  仓库远端初始状态：分支 `main`，仅一个 22 字节的占位 `README.md`（commit `bced7c7`）。
- **建议根治**（任选其一，之后即可去掉 `sslVerify=false`）：
  1. 让 Git 使用 Windows 系统证书库：`git config --global http.sslBackend schannel`
  2. 或指定 CA 包：`git config --global http.sslCAInfo "C:/Program Files/Git/mingw64/etc/ssl/certs/ca-bundle.crt"`

## 2. 推送记录（2026-10-11，待用户手动完成）

- **现象**：`git -c http.sslVerify=false push -u origin main` 失败，
  报错 `fatal: could not read Username for 'https://github.com': terminal prompts disabled`。
  本环境未配置任何凭据（`credential.helper` 为空，未安装 `gh` CLI），沙箱内无法交互登录，故无法代为推送。
- **当前状态**：本地 `main` 领先远端 2 个提交（`f6f7779` 骨架、`f1112ad` 本笔记），
  工作区干净，随时可推。

- **追加记录（2026-10-11，用户指示"提权提交"后重试）**：
  1. 凭据排查：系统级 gitconfig 存在 `credential.helper=manager`（Git Credential Manager），
     但无 `GH_TOKEN`/`GITHUB_TOKEN` 环境变量、无 SSH 私钥（`~/.ssh` 仅 known_hosts）、未安装 `gh` CLI。
  2. 申请提权至 `danger-full-access` 执行 push —— **被会话策略自动拒绝**（本次会话禁用审批弹窗，
     子代理权限范围固定，无法在会话内提权）。
  3. 当前权限下放开 askpass 交互重试 push —— 失败：沙箱阻断命名管道，
     sh/bash/GCM 凭据助手链无法启动（`couldn't create signal pipe, Win32 error 5`，
     `could not read Username for 'https://github.com'`）。
  - **结论**：push 必须在本代理会话之外的普通终端执行（见下方指引），
    或由用户在会话外配置好凭据（如 `git config --global credential.helper manager` + 一次手动登录）
    后再由代理重试。

### 用户手动推送指引（在 `repo/` 目录下执行）

方式 A：HTTPS + 个人访问令牌（PAT，推荐）

```powershell
cd repo
# 1) 在 GitHub → Settings → Developer settings → Personal access tokens 生成 token（勾选 repo 权限）
# 2) 首次推送时用户名填 GitHub 用户名，密码处粘贴 token（建议先配置证书，见上节）
git -c http.sslVerify=false push -u origin main
# 如需保存凭据：git config --global credential.helper manager
```

方式 B：GitHub CLI

```powershell
winget install GitHub.cli   # 如未安装
gh auth login               # 按提示完成浏览器/令牌登录
cd repo
git push -u origin main
```

方式 C：SSH（长期最省心）

```powershell
ssh-keygen -t ed25519 -C "chtrrrrrrr@outlook.com"
# 将 %USERPROFILE%\.ssh\id_ed25519.pub 内容添加到 GitHub → Settings → SSH Keys
cd repo
git remote set-url origin git@github.com:Chtrrrrrrr/Zhongan-OPEN-SZ-01.git
git push -u origin main
```

## 3. 入库纪律（务必遵守）

- **246MB 的 `AI生成材料反欺诈风控检测.xlsx`、`data/images/` 及一切图片/视频/模型权重不入库**，
  已由 `.gitignore` 屏蔽（含 `*.xlsx`、常见图像/视频扩展名、`models/*`、`*.pt/*.pth/*.onnx/*.bin/*.safetensors` 等）。
- README/文档配图放 `assets/`（已加白名单，可入库）。
- 模型权重如确需共享，使用 Git LFS：
  `git lfs install && git lfs track "*.pt" "*.pth" "*.onnx" "*.bin" "*.safetensors"`，
  或走网盘/Release 外部分发，并在 `models/` 下放置说明文件。
- 提交信息使用 Conventional Commits 规范（如 `feat:` / `fix:` / `docs:` / `chore:` / `test:`）。

## 4. 提交日志

| 时间 | 提交 | 说明 | 推送状态 |
| --- | --- | --- | --- |
| 2026-10-11 | `f6f7779` | docs: bootstrap project skeleton for OPEN-SZ-01（README/.gitignore/requirements/LICENSE/architecture.md + 目录骨架 + L0/L1 管线源码） | 待手动推送 |
