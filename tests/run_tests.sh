#!/bin/bash
# 测试运行脚本
# 自动使用虚拟环境运行测试

# Get the directory where the script is located (tests/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of tests/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"
VENV_PYTEST="$PROJECT_ROOT/.venv/bin/pytest"

# 检查虚拟环境是否存在
if [ ! -f "$VENV_PYTEST" ]; then
    echo "❌ 错误: 虚拟环境不存在或 pytest 未安装"
    echo "请确保 .venv 目录存在且已安装所有依赖"
    exit 1
fi

# 检查 beanie 是否可用
if ! $VENV_PYTHON -c "import beanie" 2>/dev/null; then
    echo "❌ 错误: beanie 模块未安装在虚拟环境中"
    echo "请运行: .venv/bin/pip install beanie"
    exit 1
fi

echo "✅ 使用虚拟环境: $VENV_PYTHON"
echo "✅ pytest 路径: $VENV_PYTEST"
echo ""

# 如果没有参数，显示帮助
if [ $# -eq 0 ]; then
    echo "用法: $0 <测试选项>"
    echo ""
    echo "示例:"
    echo "  $0 test_memcell_crud_complete.py -v -s"
    echo "  $0 test_memcell_crud_complete.py::TestBasicCRUD -v -s"
    echo "  $0 test_memcell_crud_complete.py::TestBasicCRUD::test_01_append_and_get_by_event_id -v -s"
    echo ""
    echo "收集测试:"
    echo "  $0 test_memcell_crud_complete.py --collect-only"
    echo ""
    echo "查看日志输出:"
    echo "  $0 test_memcell_crud_complete.py -v -s --log-cli-level=INFO"
    echo ""
    exit 0
fi

# 处理测试路径参数
# 如果第一个参数不是以 tests/ 开头，自动添加 tests/ 前缀
FIRST_ARG="$1"
if [[ "$FIRST_ARG" != tests/* ]] && [[ "$FIRST_ARG" != -* ]] && [[ -n "$FIRST_ARG" ]]; then
    # 第一个参数不是选项（不是以 - 开头），且不是以 tests/ 开头
    shift
    TEST_PATH="tests/$FIRST_ARG"
    ARGS="$TEST_PATH $@"
else
    # 参数已经包含 tests/ 或者是选项
    ARGS="$@"
fi

# 运行 pytest (从项目根目录运行，以确保正确的导入路径)
echo "运行测试..."
echo "命令: cd $PROJECT_ROOT && $VENV_PYTEST $ARGS"
echo ""
cd "$PROJECT_ROOT" && exec $VENV_PYTEST $ARGS
