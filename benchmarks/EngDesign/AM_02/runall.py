import subprocess


scripts = [
    "./test_eval_o1.py",
    "./test_eval_o3_high.py",
    "./test_eval_o3.py",
    "./test_eval_o4_mini_high.py",
    "./test_eval_o4_mini.py",

]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {script}:")
        print(result.stderr)
        break  # 出错时停止执行
    else:
        print(f"{script} completed successfully.")
        print(result.stdout)
