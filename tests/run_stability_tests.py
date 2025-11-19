#!/usr/bin/env python3
"""
稳定性测试运行脚本

用于运行 memsys 系统的稳定性测试套件
"""
import os
import sys
import asyncio
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# 设置测试环境变量
os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("DB_POOL_SIZE", "5")
os.environ.setdefault("DB_MAX_OVERFLOW", "3")


class StabilityTestRunner:
    """稳定性测试运行器"""

    def __init__(self, verbose: bool = False, parallel: bool = False):
        self.verbose = verbose
        self.parallel = parallel
        self.test_results = {}
        self.start_time = None
        self.end_time = None

    def run_test_suite(self, test_suite: str = "all") -> Dict[str, Any]:
        """运行测试套件"""
        print(f"🚀 开始运行稳定性测试套件: {test_suite}")
        self.start_time = time.time()

        test_suites = {
            "database": self._run_database_tests,
            "concurrency": self._run_concurrency_tests,
            "integration": self._run_integration_tests,
            "all": self._run_all_tests,
        }

        if test_suite not in test_suites:
            raise ValueError(f"未知的测试套件: {test_suite}")

        try:
            results = test_suites[test_suite]()
            self.end_time = time.time()
            self._print_summary(results)
            return results
        except Exception as e:
            print(f"❌ 测试运行失败: {str(e)}")
            return {"error": str(e)}

    def _run_database_tests(self) -> Dict[str, Any]:
        """运行数据库稳定性测试"""
        print("\n📊 运行数据库稳定性测试...")
        return self._run_pytest_tests("tests/test_stability_database.py", "数据库")

    def _run_concurrency_tests(self) -> Dict[str, Any]:
        """运行并发稳定性测试"""
        print("\n⚡ 运行并发稳定性测试...")
        return self._run_pytest_tests("tests/test_stability_concurrency.py", "并发")

    def _run_integration_tests(self) -> Dict[str, Any]:
        """运行集成稳定性测试"""
        print("\n🔗 运行集成稳定性测试...")
        return self._run_pytest_tests("tests/test_stability_integration.py", "集成")

    def _run_all_tests(self) -> Dict[str, Any]:
        """运行所有稳定性测试"""
        print("\n🎯 运行所有稳定性测试...")

        all_results = {}

        # 运行各个测试套件
        test_suites = [
            ("database", self._run_database_tests),
            ("concurrency", self._run_concurrency_tests),
            ("integration", self._run_integration_tests),
        ]

        for suite_name, suite_func in test_suites:
            try:
                results = suite_func()
                all_results[suite_name] = results
            except Exception as e:
                all_results[suite_name] = {"error": str(e)}
                print(f"❌ {suite_name} 测试套件失败: {str(e)}")

        return all_results

    def _run_pytest_tests(self, test_file: str, test_name: str) -> Dict[str, Any]:
        """运行pytest测试"""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_file,
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--durations=10",
        ]

        if self.parallel:
            cmd.extend(["-n", "auto"])

        print(f"执行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=project_root
            )

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _print_summary(self, results: Dict[str, Any]):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("📋 稳定性测试摘要")
        print("=" * 60)

        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            print(f"⏱️  总执行时间: {total_time:.2f}秒")

        if isinstance(results, dict) and "error" in results:
            print(f"❌ 测试失败: {results['error']}")
            return

        if isinstance(results, dict) and all(
            isinstance(v, dict) for v in results.values()
        ):
            # 多个测试套件的结果
            for suite_name, suite_results in results.items():
                self._print_suite_summary(suite_name, suite_results)
        else:
            # 单个测试套件的结果
            self._print_suite_summary("测试套件", results)

        print("=" * 60)

    def _print_suite_summary(self, suite_name: str, results: Dict[str, Any]):
        """打印测试套件摘要"""
        print(f"\n📊 {suite_name} 测试结果:")

        if "error" in results:
            print(f"  ❌ 错误: {results['error']}")
            return

        if "success" in results:
            status = "✅ 通过" if results["success"] else "❌ 失败"
            print(f"  {status}")

        if "stdout" in results and results["stdout"]:
            # 解析pytest输出
            lines = results["stdout"].split('\n')
            for line in lines:
                if "passed" in line or "failed" in line or "error" in line:
                    print(f"    {line.strip()}")

        if "stderr" in results and results["stderr"]:
            print(f"  ⚠️  警告/错误: {results['stderr'][:200]}...")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行 memsys 稳定性测试")
    parser.add_argument(
        "--suite",
        choices=["database", "concurrency", "integration", "all"],
        default="all",
        help="要运行的测试套件",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--parallel", "-p", action="store_true", help="并行运行测试")
    parser.add_argument("--output", "-o", help="输出结果到文件")

    args = parser.parse_args()

    # 创建测试运行器
    runner = StabilityTestRunner(verbose=args.verbose, parallel=args.parallel)

    # 运行测试
    results = runner.run_test_suite(args.suite)

    # 输出结果到文件
    if args.output:
        import json

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📄 测试结果已保存到: {args.output}")

    # 返回适当的退出码
    if isinstance(results, dict) and "error" in results:
        sys.exit(1)

    if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
        # 多个测试套件
        all_success = all(
            suite_results.get("success", False)
            for suite_results in results.values()
            if "error" not in suite_results
        )
        sys.exit(0 if all_success else 1)
    else:
        # 单个测试套件
        success = results.get("success", False)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
