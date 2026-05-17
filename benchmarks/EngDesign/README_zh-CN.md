# EngDesign

该目录下的任务均来自于 [EngDesign](https://github.com/AGI4Engineering/EngDesign/tree/main) 筛选了该项目中的相对复杂度较高，接近工程实践的任务
```
`CY_03`
`WJ_01`
`XY_05`
`AM_02`
`AM_03`
`YJ_02`
`YJ_03`
```

## 环境配置
### 1. 安装并登录 Docker
请在 [hub.docker.com](https://hub.docker.com/) 注册并验证您的电子邮件。
在您的计算机上下载并安装 [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
启动 Docker Desktop 并登录您的帐户。
请确保 Docker Desktop 可以访问您的驱动器（检查设置）

### 2. 通过 CLI 进行身份验证

在终端中运行：

   ```bash
   docker login -u your_dockerhub_username
   ```

### 3. 构建 Docker 镜像

在 Frontier-Engineering 仓库根目录下运行以下命令：

   ```bash
   docker build -t engdesign-sim -f benchmarks/EngDesign/Dockerfile benchmarks/EngDesign
   ```

或者在当前 `benchmarks/EngDesign` 目录下运行：

   ```bash
   docker build -t engdesign-sim .
   ```

### 4. 启动 Docker 容器

挂载本地项目目录并在容器中启动 bash 会话：

   ```bash
   docker run -it --rm -v /path/to/your/local/directory:/app --entrypoint bash engdesign-sim
   ```

## 评测方法
```
export ENGDESIGN_EVAL_MODE=docker
export ENGDESIGN_DOCKER_IMAGE=engdesign-sim
python -m frontier_eval task=engdesign algorithm=openevolve algorithm.iterations=10
```

`task=engdesign` 会加载 `frontier_eval/conf/task/engdesign.yaml`，并映射到大小写敏感的
`benchmarks/EngDesign`。如果直接使用通用 unified 任务，应写
`task.benchmark=EngDesign`，不要写成 `engdesign`。
